"""ramjet test: the tests under the paths you name that the GPUs you name can run.
Each test file is one call to a Modal Function, which Modal spreads over as many
containers as it gives you, each taking as many calls at once as it has cores. A
container collects the file's tests and leaves out those whose tci constraints
(torch/testing/_internal/tci) its machine doesn't meet."""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import modal

from tools.ramjet import sync, worker
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


def collect(
    image: Build,
    commit: str,
    resources: Resources,
    names: list[str],
    pytest_args: list[str],
    ramjet_dir: Path,
) -> list[worker.Collected | Exception]:
    """Each test file's tests, in the order of names, collected by containers that
    boot from the build image and take your changes since its commit. A call that
    Modal lost comes back as its exception."""
    changes = sync.diff(image.commit, commit)
    changed = [path for path, status in changes.items() if status != "D"]
    deleted = [path for path, status in changes.items() if status == "D"]
    with (
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
            return list(calls)


def list_tests(
    image: Build,
    commit: str,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    ramjet_dir: Path,
) -> int:
    """Print the tests asked for that these containers can run, one JSON object per
    line, and return the exit code: 1 if a file's tests couldn't be collected."""
    on = resources.gpu or "a CPU container"
    log.info("collecting %s on %s", count(len(names), "file"), on)
    results = collect(image, commit, resources, list(names), pytest_args, ramjet_dir)
    listed = failed = 0
    left_out: Counter[str] = Counter()  # tests, by the tci constraint that fails
    for (name, asked), result in zip(names.items(), results):
        if isinstance(result, Exception):  # Modal lost the call
            error = repr(result)
        else:
            took = result["seconds"], result["container"]
            log.debug("test/%s.py took %.1f s on container %s", name, *took)
            error = result["error"].rstrip()
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
                    print(json.dumps({"nodeid": nodeid}))
                    listed += 1
        if error:
            log.error("collecting test/%s.py failed:\n%s", name, error)
            failed += 1
    files = count(len(names) - failed, "file")
    log.info("listed %s from %s", count(listed, "test"), files)
    if left_out:
        tests = count(left_out.total(), "test")
        log.info("left out %s that %s can't run:", tests, on)
        for unmet, n in left_out.most_common():
            log.info("  %d with %s", n, unmet)
    return 1 if failed else 0
