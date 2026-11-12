"""What ramjet test shows while its containers collect and run. On a terminal it is a
block under the log's lines, as wide as the terminal and drawn anew in place: a bar,
a row of facts and, while tests run, a row for each container. Elsewhere it is a line
of the log when there is news. Both show what a Collecting or a Running knows, which
test.py keeps current."""

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Protocol, TextIO

from tools.ramjet.common import count
from tools.ramjet.logger import bold, dim, green, log, MARGIN, red, terminal, through
from tools.ramjet.worker import Collected, Ran, Test


COLUMNS = 80  # what the lines keep to, as the rest of the log does
WIDTH = COLUMNS - MARGIN  # of the strip that says where a run's seconds went
# The bar's cells. Flat: not started. Half a cell: done. In between: running, and
# higher once half of its tests have ended
TODO, RISING, DONE = "▁", "▂▃", "▄"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # beside the time: it turns while containers work
FRAME_SECONDS = 0.09  # between two drawings of the block
POLL_SECONDS = 2  # between two questions to Modal
ROWS = 8  # of containers at most, and fewer in a short terminal
SLOW = 1.5  # times its plan: what the time left allows the newest piece
TIP_SECONDS = 180  # without a container, before the block says what you can do
# Where the log is not on a terminal: a line after QUIET_SECONDS without one, and no
# two within APART_SECONDS
QUIET_SECONDS = 300
APART_SECONDS = 15
# Modal's GPUs that PyTorch's CI tests on as well, for the tip to name
CI_GPUS = ("L4", "A10", "T4")
# What a run's seconds go to: "waiting" while there is no container
DOING = ("waiting", "collecting", "running")
# What a terminal takes (ECMA-48, and xterm's control sequences)
UP = "\r\033[{}A"  # to the start of the row so many rows up
END = "\033[K\n"  # erase the rest of the row, and go to the next
BELOW = "\033[J"  # erase from here down
WRAP = "\033[?7{}"  # h: a row too long goes on in the next, l: it is cut
CURSOR = "\033[?25{}"  # h: shown, l: hidden
TITLE = "\033]2;{}\a"  # name the window
TITLES = "\033[{};2t"  # 22: keep the window's name, 23: bring it back

Style = Callable[[str], str]
Row = list[tuple[str, Style | None]]  # a row's text, in parts that may have a style


def span(seconds: float) -> str:
    """A length of time in its largest units: 40 s, 6 min, 1 h 12 min."""
    if round(seconds) < 60:
        return f"{max(round(seconds), 1)} s"
    hours, minutes = divmod(round(seconds / 60), 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def clock(seconds: float) -> str:
    """Minutes and seconds, as a timer shows them: 2:29."""
    return "{}:{:02}".format(*divmod(int(seconds), 60))


def paint(row: Row) -> str:
    return "".join(style(text) if style else text for text, style in row)


def spread(left: list[Row], right: list[Row], width: int) -> Row:
    """A row so wide with facts on its left and facts on its right. Those on the
    right give way, the last first, until all fit."""
    while True:
        sides = [
            [part for fact in side for part in [(" • ", dim), *fact]][1:]
            for side in (left, right)
        ]
        gap = width - sum(len(text) for side in sides for text, _ in side)
        if gap >= 2 or not right:
            return [*sides[0], (" " * max(gap, 0), None), *sides[1]]
        right = right[:-1]


def bar(done: float, running: list[tuple[float, float]], width: int) -> Row:
    """The bar, so many cells wide: the share of the work that is done, then what
    runs now, each by its share of the work and how far along it is, the furthest
    first."""
    full = min(int(done * width + 1e-9), width)
    share = sum(part for part, _ in running)
    # Up to where done and running reach together: rounding each on its own could
    # leave a cell flat although all the work has started
    upto = min(max(round((done + share) * width), full + 1), width)
    lit = upto - full if running else 0
    furthest = sorted(running, key=lambda piece: -piece[1])
    cells = ""
    for cell in range(lit):
        at, far = (cell + 0.5) / lit * share, furthest[-1][1]
        for part, progress in furthest:  # the one this cell is in
            if at <= part:
                far = progress
                break
            at -= part
        cells += RISING[far >= 0.5]
    return [(DONE * full + cells, None), (TODO * (width - full - lit), dim)]


@dataclass
class Meter:
    """A run's containers: how many Modal says there are, what they have billed, and
    where the run's seconds went."""

    gpu: str | None  # each container's, as --gpu names them
    rate: Decimal  # dollars a container bills an hour
    began: float = field(default_factory=time.monotonic)
    asked: float = field(default_factory=time.monotonic)  # when Modal last said
    containers: int = 0
    billed: float = 0.0  # container-seconds
    spent: dict[str, float] = field(default_factory=lambda: dict.fromkeys(DOING, 0.0))

    def told(self, containers: int, doing: str) -> None:
        """Modal says how many containers there are now: until now there were as many
        as it said last, doing this."""
        now = time.monotonic()
        seconds, self.asked = now - self.asked, now
        self.billed += self.containers * seconds
        self.spent[doing if self.containers else "waiting"] += seconds
        self.containers = containers

    def cost(self, now: float) -> str:
        """What the containers have billed so far, to the cent."""
        seconds = self.billed + self.containers * (now - self.asked)
        return f"${self.rate * Decimal(seconds) / 3600:.2f}"

    def named(self, containers: int) -> str:
        """So many containers, by their GPUs if they have any: 2 L4, no L4."""
        if not containers:
            return f"no {self.gpu or 'container'}"
        if self.gpu is None:
            return count(containers, "container")
        return f"{containers} {self.gpu}"

    def of(self, wanted: int) -> str:
        """The containers there are, of those the run can use now: 7/141 L4. One
        with nothing left to take is about to stop, and isn't counted."""
        if not wanted:
            return self.named(self.containers)
        wants = f"{wanted:,} {self.gpu}" if self.gpu else count(wanted, "container")
        return f"{min(self.containers, wanted):,}/{wants}"

    def where(self) -> list[Row]:
        """Where the run's seconds went: the bar's width shared among them, and under
        it what its cells stand for."""
        whole = sum(self.spent.values()) or 1.0
        waiting = round(WIDTH * self.spent["waiting"] / whole)
        collecting = round(WIDTH * self.spent["collecting"] / whole)
        strip: Row = [(TODO * waiting, dim), (RISING[0] * collecting, None)]
        strip.append((DONE * (WIDTH - waiting - collecting), None))
        names = {"waiting": f"with {self.named(0)}"}
        said = [
            f"{span(seconds)} {names.get(doing, doing)}"
            for doing, seconds in self.spent.items()
            if seconds
        ]
        return [strip, [(" • ".join(said), dim)]]


@dataclass
class Collecting:
    """What ramjet knows while test files are collected."""

    meter: Meter
    files: int
    cores: int  # files a container collects at once
    back: list[Collected | Exception] = field(default_factory=list)  # as they come

    def working(self, now: float) -> bool:
        return self.meter.containers > 0

    def facts(self, now: float) -> tuple[list[Row], list[Row]]:
        """What to say of the files: on the block's left, and on its right."""
        meter = self.meter
        if not meter.containers and not self.back:
            waited = f"Modal has given {meter.named(0)} in {span(now - meter.began)}"
            bill = meter.cost(now) if meter.billed else "nothing billed yet"
            return [[(waited, None)]], [[(bill, dim)]]
        got = [file for file in self.back if not isinstance(file, Exception)]
        left: list[Row] = [[(f"{len(self.back)} of {count(self.files, 'file')}", None)]]
        if got:
            left.append([(count(sum(len(file["ids"]) for file in got), "test"), dim)])
        calls = -(-(self.files - len(self.back)) // self.cores)  # a container each
        right: list[Row] = [[(meter.of(calls), None)]]
        if got and meter.containers:  # at the files' mean so far
            each = sum(file["seconds"] for file in got) / len(got)
            at_once = self.cores * meter.containers
            togo = (self.files - len(self.back)) * each / at_once
            right.append([(f"{span(togo)} left", None)])
        return left, [*right, [(meter.cost(now), dim)]]

    def rows(self, now: float, width: int) -> list[Row]:
        """The block, so wide: its bar, its facts and, after a long wait that has
        cost nothing, what you can do about it."""
        meter = self.meter
        flying = min(self.files - len(self.back), self.cores * meter.containers)
        running = [(flying / self.files, 0.0)] if flying else []
        done = len(self.back) / self.files
        rows = [bar(done, running, width), spread(*self.facts(now), width)]
        idle = not meter.containers and not self.back and not meter.billed
        if idle and now - meter.began >= TIP_SECONDS:
            others = [gpu for gpu in CI_GPUS if gpu != meter.gpu]
            tip = f"try --gpu {' or '.join(others)}: " if meter.gpu in CI_GPUS else ""
            rows.append([(f"{tip}Ctrl-C is free while nothing runs", dim)])
        return rows

    def news(self) -> tuple[int, int]:
        """What a new line is for when it grows: the containers, and the tenths of
        the files that are back."""
        return self.meter.containers, 10 * len(self.back) // self.files

    def title(self) -> str:
        if not self.meter.containers and not self.back:
            return f"ramjet: {self.meter.named(0)} yet"
        return "ramjet: collecting"


class Planned(Protocol):
    """A piece, as test.py cuts it."""

    seconds: float
    tests: dict[str, list[str]]
    times: dict[str, float]


@dataclass(frozen=True)
class Flight:
    """A container on a piece, as it last said."""

    piece: int  # its number, or 0 once another container started it
    since: float  # when the container said it started
    done: int  # tests of the piece that have ended
    test: Test | None  # the one that runs


@dataclass
class Running:
    """What ramjet knows while pieces run."""

    meter: Meter
    pieces: Sequence[Planned]  # a piece's number is its place here, from 1
    back: list[Ran | Exception] = field(default_factory=list)  # as they come
    started: dict[str, Flight] = field(default_factory=dict)  # by Modal's container id
    failed: int = 0  # tests of the pieces that are back
    sizes: list[int] = field(init=False)  # how many tests each piece has

    def __post_init__(self) -> None:
        self.sizes = [sum(map(len, piece.tests.values())) for piece in self.pieces]

    def report(self, piece: int, container: str, done: int, test: Test | None) -> None:
        """A container says how far it is on a piece, which no other one runs any
        more."""
        last = self.started.get(container)
        since = last.since if last and last.piece == piece else time.monotonic()
        moved = {
            other: replace(flight, piece=0)
            for other, flight in self.started.items()
            if flight.piece == piece and other != container
        }
        mine = {container: Flight(piece, since, done, test)}
        self.started = {**self.started, **moved, **mine}  # anew: other threads read

    def returned(self) -> dict[int, float]:
        """The seconds each piece that is back took, by its number."""
        return {
            ran["piece"]: ran["seconds"]
            for ran in self.back
            if not isinstance(ran, Exception)
        }

    def flying(self) -> list[Flight]:
        """The containers on a piece that isn't back, the longest on its piece
        first."""
        back = self.returned()
        on = [f for f in self.started.values() if f.piece and f.piece not in back]
        return sorted(on, key=lambda flight: flight.since)

    def working(self, now: float) -> bool:
        return bool(self.flying())

    def tests(self) -> tuple[int, int]:
        """How many tests have ended: those of the pieces that are back, and those
        the containers say have on the pieces they run. And how many there are."""
        back = sum(self.sizes[piece - 1] for piece in self.returned())
        flying = sum(min(f.done, self.sizes[f.piece - 1]) for f in self.flying())
        return back + flying, sum(self.sizes)

    def left(self, now: float) -> float | None:
        """The seconds to go, or None when there is no telling: the work left, at
        the pace of the pieces that are back, shared by the containers."""
        planned = [piece.seconds for piece in self.pieces]
        took = self.returned()
        flying = [(flight.piece, now - flight.since) for flight in self.flying()]
        # The first pieces back are the quick ones, so the plan weighs as four pieces
        prior = 4 * sum(planned) / len(planned)
        done = sum(planned[piece - 1] for piece in took)
        pace = (sum(took.values()) + prior) / (done + prior)
        ahead = sum(planned) - done - sum(planned[piece - 1] for piece, _ in flying)
        going = sum(max(planned[piece - 1] * pace - on, 0) for piece, on in flying)
        # No sooner than the newest piece would end if it turned out slow
        tail = max((SLOW * planned[piece - 1] - on for piece, on in flying), default=0)
        waiting = len(planned) - len(self.back) - len(flying)
        if not self.meter.containers or (waiting <= 0 and tail <= 0):
            return None
        return max((ahead * pace + going) / self.meter.containers, tail)

    def facts(self, now: float) -> tuple[list[Row], list[Row]]:
        """What to say of the tests: on the block's left, and on its right."""
        meter, (ended, tests) = self.meter, self.tests()
        left: list[Row] = [[(f"{ended:,} of {count(tests, 'test')}", None)]]
        if self.failed:
            left.append([("✗", red), (f" {self.failed:,} failed", bold)])
        # A piece runs in a container of its own
        right: list[Row] = [[(meter.of(len(self.pieces) - len(self.back)), None)]]
        if (togo := self.left(now)) is not None:  # without a countdown at the end
            right.append([(f"{span(togo) if togo >= 60 else '< 1 min'} left", None)])
        return left, [*right, [(meter.cost(now), dim)]]

    def rows(self, now: float, width: int) -> list[Row]:
        """The block's bar and its facts, so wide. A piece that runs is as far along
        as its container says."""
        planned = [piece.seconds for piece in self.pieces]
        whole = sum(planned)
        done = sum(planned[piece - 1] for piece in self.returned()) / whole
        running = [
            (planned[f.piece - 1] / whole, f.done / self.sizes[f.piece - 1])
            for f in self.flying()
        ]
        return [bar(done, running, width), spread(*self.facts(now), width)]

    def wide(self, now: float, most: int, width: int) -> list[Row]:
        """A row as wide as the terminal for each container on a piece, up to the
        most and the longest on its piece first: how long it has been on it, how
        many of the piece's tests have ended, and the test that runs. A name that
        doesn't fit is cut."""
        flying = self.flying()
        shown = flying[:most]
        if not shown:
            return []
        said = [
            (clock(now - f.since), str(f.done), str(self.sizes[f.piece - 1]))
            for f in shown
        ]
        # As wide in every row, so that the names start in one column. The tests
        # that have ended get the width of the tests there are, which they grow to
        on, _, size = (max(map(len, column)) for column in zip(*said))
        rows: list[Row] = []
        for flight, (time_on, ended, tests) in zip(shown, said):
            if flight.test:  # torchci's file is a path: test/test_nn.py
                file, suite, case = flight.test
                name = "::".join(filter(None, (Path(file).name, suite, case)))
            else:  # none yet: the piece's first file, and how many more it has
                files = list(self.pieces[flight.piece - 1].tests)
                rest = f" +{len(files) - 1}" if len(files) > 1 else ""
                name = f"{Path(files[0]).name}.py{rest}"
            start = f" {time_on:>{on}} "
            progress = f"[{ended:>{size}}/{tests:>{size}}]"
            room = width - len(start) - len(progress) - 2  # a space on each side
            if len(name) > room:
                name = f"{name[: max(room - 1, 0)]}…"
            rows.append([(start, None), (progress, dim), (f" {name}", None)])
        if 0 < most < len(flying):
            others = max(self.meter.containers, len(flying)) - most
            more = count(others, "more container")
            rows.append([(f"{'':{on + 2 * size + 6}}… and {more}", dim)])
        return rows

    def news(self) -> tuple[int, int]:
        """What a new line is for when it grows: the containers, and the tenths of
        the tests that have ended."""
        ended, tests = self.tests()
        return self.meter.containers, 10 * ended // tests

    def title(self) -> str:
        ended, tests = self.tests()
        failed = f" • {self.failed:,} failed" if self.failed else ""
        return f"ramjet {100 * ended // tests}%{failed}"


def drawn(
    state: Collecting | Running, now: float, turn: int, most: int, width: int
) -> list[str]:
    """The block's rows, as a terminal so wide takes them. The first starts as the
    log's lines do, with a mark in the level's place that turns while containers
    work. The bar and the facts go from the log's margin to a space short of the
    terminal's end, where the rows of the containers end, at most so many."""
    first, *rest = state.rows(now, max(width - MARGIN - 1, 1))
    mark = green(SPINNER[turn % len(SPINNER)]) if state.working(now) else dim("·")
    start = dim(f"ramjet {time.strftime('%H:%M:%S')}".ljust(MARGIN - 2))
    under = [" " * MARGIN + paint(row) for row in rest]
    if isinstance(state, Running):
        under += [paint(row) for row in state.wide(now, most, width)]
    return [f"{start}{mark} {paint(first)}", *under]


def line(state: Collecting | Running, now: float) -> str:
    """The block's facts as a line of the log, without the last ones if it would
    pass COLUMNS."""
    left, right = state.facts(now)
    said = ["".join(text for text, _ in fact) for fact in left + right]
    while len(said) > 1 and MARGIN + 2 + len(" • ".join(said)) > COLUMNS:
        said.pop()
    return " • ".join(said)


class Block:
    """Rows of a terminal that stay under the log's lines: a line written through the
    block goes above them."""

    def __init__(self, terminal: TextIO) -> None:
        self.terminal = terminal
        self.rows: list[str] = []
        self.title = ""
        self.lock = threading.Lock()

    def __enter__(self) -> "Block":
        self.terminal.write(CURSOR.format("l") + TITLES.format(22))
        return self

    def __exit__(self, *error: object) -> None:
        self.draw([], self.title)
        self.terminal.write(TITLES.format(23) + CURSOR.format("h"))
        self.terminal.flush()

    def room(self) -> tuple[int, int]:
        """How many rows of containers fit, as the block takes half the terminal at
        most, and how wide a row can be."""
        try:
            columns, lines = os.get_terminal_size(self.terminal.fileno())
        except OSError:
            columns, lines = COLUMNS, 24
        return max(min(lines // 2 - 3, ROWS), 0), columns

    def draw(self, rows: list[str], title: str, above: str = "") -> None:
        """Draw the rows where the last ones are, under the text if there is any,
        and name the window."""
        with self.lock:
            if (rows, title, above) == (self.rows, self.title, ""):
                return
            up = UP.format(len(self.rows)) if self.rows else ""
            named = TITLE.format(title) if title != self.title else ""
            self.rows, self.title = rows, title
            cut = WRAP.format("l") + "".join(row + END for row in rows)
            text = up + named + above.replace("\n", END) + cut + WRAP.format("h")
            self.terminal.write(text + BELOW)
            self.terminal.flush()

    def write(self, text: str) -> None:
        self.draw(self.rows, self.title, text)

    def flush(self) -> None:
        self.terminal.flush()


@contextlib.contextmanager
def shown(state: Collecting | Running, ask: Callable[[], None]) -> Iterator[None]:
    """Until the block ends, ask Modal every POLL_SECONDS, which keeps the state
    current, and show the state: as a block on the log's terminal or, without one, as
    a line of the log when there is news."""
    done = threading.Event()
    screen = terminal()

    def poll() -> None:
        said, seen = time.monotonic(), (0, 0)
        while not done.is_set():
            done.wait(POLL_SECONDS)  # and once more as the block ends
            try:
                ask()
            except Exception as error:  # Modal didn't say: it may the next time
                log.debug("no word from Modal: %r", error)
            quiet, news = time.monotonic() - said, state.news()
            more = any(new > old for new, old in zip(news, seen))
            if screen or done.is_set() or quiet < APART_SECONDS:
                continue
            if more or quiet >= QUIET_SECONDS:
                log.info("  %s", line(state, time.monotonic()))
                said, seen = time.monotonic(), news

    def draw(block: Block) -> None:
        turn = 0
        while not done.wait(FRAME_SECONDS):
            rows = drawn(state, time.monotonic(), turn, *block.room())
            block.draw(rows, state.title())
            turn += 1

    threads = [threading.Thread(target=poll, name="ramjet-progress", daemon=True)]
    with contextlib.ExitStack() as stack:
        if screen:
            block = stack.enter_context(Block(screen))
            stack.enter_context(through(block))
            threads.append(threading.Thread(target=draw, args=(block,), daemon=True))
        for thread in threads:
            thread.start()
        try:
            yield
        finally:
            done.set()
            for thread in threads:
                thread.join()
