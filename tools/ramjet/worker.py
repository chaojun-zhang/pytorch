"""The class ramjet test runs on Modal. Each container boots from the build image,
brings its /pytorch to your working tree once, and then serves calls. It is imported
in the containers, so it uses only the standard library and modal."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import modal


ROOT = "/pytorch"
# Your changes since the build image's commit, where test.py mounts them
CHANGES = "/tmp/ramjet/changes.tar"
DELETED = "/tmp/ramjet/deleted"  # paths, NUL-separated
TAIL = 2000  # characters of a failed collect's output to return


class Collected(TypedDict):
    ids: list[str]  # as pytest prints them: test/test_nn.py::TestNN::test_linear
    error: str  # the end of the output if collecting failed, else empty
    seconds: float  # how long collecting took
    container: str  # Modal's id for the container that did it


class Worker:
    @modal.enter()
    def sync(self) -> None:
        """Once per container, what run.sync_files does to a runner: rm for the files
        deleted, then a tar of those changed."""
        for path in filter(None, Path(DELETED).read_text().split("\0")):
            Path(ROOT, path).unlink(missing_ok=True)
        if Path(CHANGES).stat().st_size:  # empty when no file changed
            subprocess.run(["tar", "-xf", CHANGES], cwd=ROOT, check=True)

    @modal.method()
    def collect(self, name: str, pytest_args: list[str]) -> Collected:
        """The tests run_test.py would run here from one of its test files."""
        # Launched as run_test.py launches it, so the file's __main__ guard decides
        # here too: pytest alone collects files that guard never lets run
        launch = [sys.executable, "-bb", f"{name}.py", "--use-pytest", "--collect-only"]
        # Last, so a -v among your arguments can't turn the ids into a tree
        command = [*launch, *pytest_args, "--verbosity=-1"]
        # Modal's PYTHONPATH holds its own packages and the tools/ it mounts to import
        # this module, which would shadow /pytorch/tools; the image sets none
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        output = {"capture_output": True, "text": True, "errors": "replace"}
        start = time.monotonic()
        done = subprocess.run(command, cwd=f"{ROOT}/test", env=env, **output)
        failed = done.returncode != 0
        prefix = f"test/{name}.py::"  # the other lines are the file's own output
        lines = [] if failed else done.stdout.splitlines()
        return {
            "ids": [line for line in lines if line.startswith(prefix)],
            "error": (done.stderr + done.stdout)[-TAIL:] if failed else "",
            "seconds": time.monotonic() - start,
            "container": os.environ.get("MODAL_TASK_ID", ""),
        }
