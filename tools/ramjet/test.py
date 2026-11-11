"""ramjet test: the tests under the paths you name that the GPUs you name can run,
in the pieces containers would run them in. Each test file is one call to a Modal
Function, which Modal spreads over as many containers as it gives you, each taking
as many calls at once as it has cores. A container collects the file's tests and
leaves out those whose tci constraints (torch/testing/_internal/tci) its machine
doesn't meet. The rest are cut into pieces of about PIECE_SECONDS by CI's time for
each test."""

import contextlib
import json
import math
import statistics
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal

from tools.ramjet import sync, times, worker
from tools.ramjet.common import repo_root
from tools.ramjet.lineage import Build
from tools.ramjet.logger import log
from tools.ramjet.run import Resources
from tools.testing.discover_tests import TESTS


# A container's resources unless --cpu and --memory say otherwise: CORES_PER_GPU per
# GPU, or as many without one, with MIB_PER_CORE of memory each
CORES_PER_GPU = 4
MIB_PER_CORE = 4096
# The least Modal allows: a container is idle only once no file is left to collect
SCALEDOWN_SECONDS = 2
PROGRESS_SECONDS = 15  # between the lines that say how many files have come back
PIECE_SECONDS = 150
# A container's seconds per second of CI, and its seconds to start a test file: fitted
# on L4 containers running test/dynamo
SLOWDOWN = 2.28
START_SECONDS = 5.4
# CI's mean for a test, for the tests of a file it has timed none of
GUESS_SECONDS = 0.3


@dataclass
class Piece:
    """What one container runs: tests of some files, each file started once."""

    seconds: float = 0.0  # what it should take
    tests: dict[str, list[str]] = field(default_factory=dict)  # node ids, by their file


def count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def names(paths: list[str]) -> dict[str, list[str]]:
    """run_test.py's name for each test file the paths cover, with what the paths
    ask of it: the file, for all its tests, or node ids in it. A path is a test
    file, a directory of them or a node id, from the directory you are in."""
    test_dir = repo_root() / "test"
    asked: dict[str, list[str]] = {}
    for path in paths:
        file, _, node = path.partition("::")
        target = Path(file).resolve()
        found: list[str] = []
        if target.is_dir() and not node and target.is_relative_to(test_dir):
            under = "" if target == test_dir else f"{target.relative_to(test_dir)}/"
            # TESTS also names C++ binaries and tests that have no file of their own
            found = [name for name in TESTS if name.startswith(under)]
            found = [name for name in found if (test_dir / f"{name}.py").is_file()]
        elif target.is_file() and target.is_relative_to(test_dir):
            name = target.relative_to(test_dir).with_suffix("").as_posix()
            found = [name] if name in TESTS else []
        if not found:
            log.error("%s is not a test file or directory that run_test.py runs", path)
            log.error("with pytest instead: python -m tools.ramjet -- pytest %s", path)
            sys.exit(1)
        for name in found:
            file_id = f"test/{name}.py"
            asked.setdefault(name, []).append(f"{file_id}::{node}" if node else file_id)
    return asked


@contextlib.contextmanager
def progress(files: int) -> Iterator[list[worker.Collected | Exception]]:
    """Until it exits, log every PROGRESS_SECONDS how many of the files have come
    back: the caller adds each one's result to the list."""
    back: list[worker.Collected | Exception] = []
    done = threading.Event()

    def report() -> None:
        while not done.wait(PROGRESS_SECONDS):
            if not back:
                log.info("  no file collected yet: waiting for containers")
                continue
            ids = {r["container"] for r in back if not isinstance(r, Exception)}
            sizes = count(files, "file"), count(len(ids), "container")
            log.info("  %d of %s collected by %s", len(back), *sizes)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        yield back
    finally:
        done.set()
        reporter.join()


def collect(
    image: Build,
    commit: str,
    resources: Resources,
    names: list[str],
    pytest_args: list[str],
    ramjet_dir: Path,
) -> Iterator[worker.Collected | Exception]:
    """Each test file's tests, in the order of names and as each comes back, collected
    by containers that boot from the build image and take your changes since its
    commit. A call that Modal lost comes back as its exception."""
    changes = sync.diff(image.commit, commit)
    changed = [path for path, status in changes.items() if status != "D"]
    deleted = [path for path, status in changes.items() if status == "D"]
    with (
        progress(len(names)) as back,
        tempfile.NamedTemporaryFile(dir=ramjet_dir, suffix=".tar") as tar,
        tempfile.NamedTemporaryFile("w", dir=ramjet_dir) as gone,
    ):
        if changed:  # given no paths, git archives the whole tree
            sync.archive(commit, changed, Path(tar.name))
        gone.write("\0".join(deleted))
        gone.flush()
        files = (
            modal.Image.from_id(image.image_id)
            # The layer Modal's builder adds: on the build image alone, which is a
            # sandbox's snapshot, a Function's container can't start
            .run_commands("true")
            # Mounted as each container starts, so runs share the image
            .add_local_file(tar.name, worker.CHANGES)
            .add_local_file(gone.name, worker.DELETED)
        )
        app = modal.App("ramjet-test")
        size = {"scaledown_window": SCALEDOWN_SECONDS, **resources._asdict()}
        # Collecting a file is one process, so a container takes one file per core
        at_once = modal.concurrent(max_inputs=resources.cpu)
        workers = app.cls(image=files, **size)(at_once(worker.Worker))
        with app.run():  # its containers stop when ramjet does, Ctrl-C included
            kwargs = {"pytest_args": pytest_args}
            calls = workers().collect.map(names, kwargs=kwargs, return_exceptions=True)
            for result in calls:
                back.append(result)
                yield result


def timed(
    kept: dict[str, list[str]], machine: dict[str, Any]
) -> dict[str, list[tuple[str, float]]]:
    """Each test with the seconds it should take in a container: SLOWDOWN times CI's
    time for it on machines with the same GPUs, else on any machine, else the median
    of its file's tests."""
    files = {f"test/{name}.py": len(ids) for name, ids in kept.items()}
    device, gpus = machine["device"], machine["gpus"]
    ci = times.fetch(files, device, gpus)
    same = guessed = 0
    estimates = {}
    for name, ids in kept.items():
        seconds = []  # CI's for each test, or None for one it has never run
        for nodeid in ids:
            here, anywhere = ci.get(nodeid, (None, None))
            seconds.append(anywhere if here is None else here)
            same += here is not None
        known = [s for s in seconds if s is not None]
        typical = statistics.median(known) if known else GUESS_SECONDS
        guessed += len(ids) - len(known)
        scaled = [SLOWDOWN * (typical if s is None else s) for s in seconds]
        estimates[name] = list(zip(ids, scaled))
    others = sum(files.values()) - same - guessed
    on = f"{gpus} x {device}" if gpus else "no GPU"
    sources = count(same, "test"), on, others, guessed
    log.debug("CI's times: %s with %s, %d on other machines, %d guessed", *sources)
    if guessed:
        log.info("guessed the time of %s that CI has none for", count(guessed, "test"))
    return estimates


def shard(timed: dict[str, list[tuple[str, float]]]) -> list[Piece]:
    """Pieces of about PIECE_SECONDS, the longest first. A file that takes longer is
    cut into runs of consecutive tests, and shorter ones share a piece: one run of a
    file per piece, so that a piece starts each of its files once."""
    runs: list[tuple[float, str, list[str]]] = []
    for name, tests in timed.items():
        seconds, ids = START_SECONDS, []
        for nodeid, estimate in tests:
            if ids and seconds + estimate > PIECE_SECONDS:
                runs.append((seconds, name, ids))
                seconds, ids = START_SECONDS, []
            seconds += estimate
            ids.append(nodeid)
        runs.append((seconds, name, ids))
    pieces: list[Piece] = []
    for seconds, name, ids in sorted(runs, key=lambda run: -run[0]):
        room = PIECE_SECONDS - seconds
        fits = (p for p in pieces if p.seconds <= room and name not in p.tests)
        piece = next(fits, None)  # the first with room, of the longest so far
        if piece is None:
            piece = Piece()
            pieces.append(piece)
        piece.seconds += seconds
        piece.tests[name] = ids
    return sorted(pieces, key=lambda piece: -piece.seconds)


def list_tests(
    image: Build,
    commit: str,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    ramjet_dir: Path,
) -> int:
    """Print the tests asked for that these containers can run, each with its piece,
    one JSON object per line, and return the exit code: 1 if a file's tests couldn't
    be collected."""
    on = resources.gpu or "a CPU container"
    log.info("collecting %s on %s", count(len(names), "file"), on)
    results = collect(image, commit, resources, list(names), pytest_args, ramjet_dir)
    failed = 0
    kept: dict[str, list[str]] = {}  # each file's tests to run
    left_out: Counter[str] = Counter()  # tests, by the tci constraint that fails
    machines: Counter[str] = Counter()  # containers, by what tci read of them
    # strict, so that collect runs to its end, where its containers stop
    for (name, asked), result in zip(names.items(), results, strict=True):
        if isinstance(result, Exception):  # Modal lost the call
            error = repr(result)
        else:
            took = result["seconds"], result["container"]
            log.debug("test/%s.py took %.1f s on container %s", name, *took)
            error = result["error"].rstrip()
            machines[json.dumps(result["machine"])] += 1
            # A node id also asks for the tests under it and for its parametrizations
            under = tuple(f"{a}{mark}" for a in asked for mark in ("::", "["))
            # A test that can run here has None; one left out, its constraint
            verdicts = {**dict.fromkeys(result["ids"]), **result["left_out"]}
            for nodeid, unmet in verdicts.items():
                if nodeid not in asked and not nodeid.startswith(under):
                    continue
                if unmet:
                    log.debug("left out %s: %s", nodeid, unmet)
                    left_out[unmet] += 1
                else:
                    kept.setdefault(name, []).append(nodeid)
        if error:
            log.error("collecting test/%s.py failed:\n%s", name, error)
            failed += 1
    files = count(len(names) - failed, "file")
    listed = sum(map(len, kept.values()))
    log.info("listed %s from %s", count(listed, "test"), files)
    if left_out:
        tests = count(left_out.total(), "test")
        log.info("left out %s that %s can't run:", tests, on)
        for unmet, n in left_out.most_common():
            log.info("  %d with %s", n, unmet)
    if len(machines) > 1:  # Modal may serve an H100 with an H200
        log.warning("containers differ: %s", ", ".join(machines))
    if kept:
        machine = json.loads(machines.most_common(1)[0][0])
        pieces = shard(timed(kept, machine))
        piece_of: dict[str, int] = {}
        for number, piece in enumerate(pieces, 1):
            ids = [nodeid for run in piece.tests.values() for nodeid in run]
            sizes = count(len(ids), "test"), count(len(piece.tests), "file")
            log.debug("piece %d, about %.0f s: %s of %s", number, piece.seconds, *sizes)
            piece_of.update(dict.fromkeys(ids, number))
        for ids in kept.values():
            for nodeid in ids:
                print(json.dumps({"nodeid": nodeid, "piece": piece_of[nodeid]}))
        minutes = math.ceil(sum(piece.seconds for piece in pieces) / 60)
        plan = count(len(pieces), "piece"), count(minutes, "container-minute")
        log.info("%s: %s, the longest about %.0f s", *plan, pieces[0].seconds)
    return 1 if failed else 0
