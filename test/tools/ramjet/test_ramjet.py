# Owner(s): ["module: devx"]

import contextlib
import dataclasses
import getpass
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfWindows,
    TestCase,
)
from torch.testing._internal.tci import machine


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.ramjet import common, inputs, lineage, sync, times


HAS_MODAL = importlib.util.find_spec("modal") is not None
if HAS_MODAL:
    import tools.ramjet.__main__ as cli
    from tools.ramjet import build, run, test, worker

sys.path.remove(str(REPO_ROOT))

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "ramjet",
    "GIT_AUTHOR_EMAIL": "ramjet@example.com",
    "GIT_COMMITTER_NAME": "ramjet",
    "GIT_COMMITTER_EMAIL": "ramjet@example.com",
}
RATES = {
    "cpu_hour_cost_sandbox": Decimal("0.1419"),
    "mem_gib_hour_cost_sandbox": Decimal("0.024"),
    "gpu_hour_cost_a10g": Decimal("1.10"),
    "gpu_hour_cost_a100_40gb": Decimal("2.10"),
    "gpu_hour_cost_a100_80gb": Decimal("2.50"),
    "gpu_hour_cost_b200": Decimal("6.25"),
    "gpu_hour_cost_h100": Decimal("3.95"),
    "gpu_hour_cost_rtx6000": Decimal("3.03"),
}
CMAKE_LISTS = """\
cmake_minimum_required(VERSION 3.18)
project(demo C)
add_library(demo demo.c)
"""
# Stands in for a test file: what collect reads is what its launcher prints, and
# what tci's plugin writes of the tests it deselects. The second test's
# parametrization is how the file was launched: its arguments and its PYTHONPATH
TEST_FILE = """\
import json
import os
import sys

print("Running 2 items in this shard")
print("test/test_file.py::TestFile::test_first")
launched = [*sys.argv[1:], os.environ.get("PYTHONPATH", "no PYTHONPATH")]
deselected = launched.pop(4).removeprefix("--tci-deselected=")
second = "test/test_file.py::TestFile::test_second"
print(f"{second}[{' '.join(launched)}]")
unmet = {"[two]": "@tci.needs(tci.gpus >= 2)", "_rocm": "@tci.only(tci.rocm)"}
with open(deselected, "w") as file:
    json.dump({second + test: constraint for test, constraint in unmet.items()}, file)
"""
# Stands in for the ci-metrics wrapper: in a directory outside the repository, it
# keeps each query it is sent, and it answers all of them with the same rows
WRAPPER = """\
#!/bin/sh
for query; do :; done
printf '%s\\n' "$query" >> {0}/queries
cat {0}/rows.json
"""
L4 = {"accelerator": "cuda", "gpus": 1, "device": "NVIDIA L4", "gpu_memory": 22.5}


@contextlib.contextmanager
def exits(test: unittest.TestCase, error: str) -> Iterator[None]:
    """The block logs the error and exits, as ramjet does on every failure."""
    with test.assertLogs("ramjet", "ERROR") as logs, test.assertRaises(SystemExit):
        yield
    test.assertIn(error, "\n".join(logs.output))


def files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


class LocalRunner:
    """Stands in for a runner sandbox: its /pytorch is a local directory, and it
    shares your filesystem, so files copied in are already there."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.filesystem = self

    def copy_from_local(self, local_path: str, remote_path: str) -> None:
        pass

    def exec(self, *args: str, workdir: str, **streams: Any) -> subprocess.Popen[str]:
        cwd = self.root / Path(workdir).relative_to("/pytorch")
        return subprocess.Popen(args, cwd=cwd, stderr=subprocess.PIPE, text=True)


@skipIfWindows(msg="ramjet runs on Linux and macOS")
class TestRepo(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)
        patcher = mock.patch.dict(os.environ, GIT_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.repo)
        common.repo_root.cache_clear()  # each test has its own repo
        self.addCleanup(common.repo_root.cache_clear)
        common.git("init", "-q", "-b", "main")
        for path in ["kept.py", "edited.py", "deleted.py", "module"]:
            self.write(path, f"{path}\n")
        common.git("add", "--all")
        common.git("commit", "-q", "-m", "base")
        self.base = common.git("rev-parse", "HEAD")

    def write(self, path: str, text: str) -> None:
        (self.repo / path).parent.mkdir(parents=True, exist_ok=True)
        (self.repo / path).write_text(text)

    def answer(self, rows: list[list[Any]]) -> None:
        """Have CI's times be these rows: a test, its seconds here and anywhere."""
        self.write(times.WRAPPER, WRAPPER.format(self.tmp))
        (self.repo / times.WRAPPER).chmod(0o755)
        (self.tmp / "rows.json").write_text(json.dumps({"rows": rows}))

    def edit(self) -> None:
        self.write("edited.py", "edited\n")
        self.write("new dir/new file.py", "added\n")
        (self.repo / "deleted.py").unlink()
        (self.repo / "module").unlink()
        self.write("module/__init__.py", "now a package\n")

    def test_snapshot_tree(self) -> None:
        index = Path(common.git("rev-parse", "--git-path", "index"))
        before = index.read_bytes()
        self.edit()
        commit = sync.snapshot_tree()
        self.assertEqual(index.read_bytes(), before)
        self.assertEqual(common.git("rev-parse", "HEAD"), self.base)
        self.assertEqual(common.git("rev-parse", f"{commit}^"), self.base)
        self.assertEqual(common.git("show", f"{commit}:new dir/new file.py"), "added")
        changes = {
            "deleted.py": "D",
            "edited.py": "M",
            "module": "D",
            "module/__init__.py": "A",
            "new dir/new file.py": "A",
        }
        self.assertEqual(sync.diff(self.base, commit), changes)
        self.assertEqual(sync.diff(commit, commit), {})

    def test_ramjet_dir(self) -> None:
        with mock.patch.object(tempfile, "tempdir", str(self.tmp)):
            path = common.ramjet_dir()
            user_dir = self.tmp / f"ramjet-{getpass.getuser()}"
            self.assertEqual(path.parent, user_dir)
            self.assertRegex(path.name, r"^repo-[0-9a-f]{8}$")
            self.assertEqual(user_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(common.ramjet_dir(), path)
            with mock.patch("os.getuid", return_value=os.getuid() + 1):
                with exits(self, "another user"):
                    common.ramjet_dir()

    def test_merge_base(self) -> None:
        common.git("remote", "add", "origin", "https://github.com/someone/pytorch.git")
        with exits(self, "no git remote"):
            sync.merge_base("HEAD")
        common.git("remote", "add", "upstream", "git@github.com:pytorch/pytorch.git")
        common.git("update-ref", "refs/remotes/upstream/main", self.base)
        self.write("edited.py", "a local commit\n")
        common.git("commit", "-q", "-am", "local")
        self.edit()
        self.assertEqual(sync.merge_base(sync.snapshot_tree()), self.base)

    def test_bundle_reaches_a_partial_clone(self) -> None:
        # Like the builder's clone of pytorch/pytorch: it has the base, but no blobs
        builder = self.tmp / "builder"
        common.git("config", "uploadpack.allowFilter", "true")
        clone = ["clone", "-q", "--filter=blob:none", "--no-checkout"]
        common.git(*clone, self.repo.as_uri(), str(builder))
        self.edit()
        commit = sync.snapshot_tree()
        bundle = sync.bundle(commit, self.base, self.tmp)
        os.chdir(builder)
        common.git("fetch", "-q", str(bundle), "refs/worktree/ramjet")
        common.git("checkout", "-q", "--force", "--detach", "FETCH_HEAD")
        self.assertEqual(files(builder), files(self.repo))

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_image_for(self) -> None:
        clones: list[bool] = []

        def fake_build(*args: Any, clone: bool) -> str:
            clones.append(clone)
            args[-1].write_text("lib.cpp\n")  # the inputs, as the builder exports them
            return f"im-{len(clones)}"

        ci_image = build.CIImage("ci:a", python="3.11")
        for patcher in [
            mock.patch.object(build, "ci_images", return_value={"13.2": ci_image}),
            mock.patch.object(build, "toolchain"),
            mock.patch.object(build, "build", side_effect=fake_build),
            mock.patch.object(sync, "merge_base", return_value=self.base),
            mock.patch.object(cli.modal.Image, "from_id"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

        def image_for(
            days_later: float = 0, python: str | None = None
        ) -> lineage.Build:
            now = time.time() + days_later * 24 * 3600
            date = f"{now:.0f} +0000"  # git's clock too, which dates the snapshot
            env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(cli.time, "time", return_value=now):
                    commit = sync.snapshot_tree()
                    return cli.image_for(None, commit, self.tmp, "13.2", python, {})

        self.write("lib.cpp", "int a;\n")
        first = image_for()
        self.assertEqual((first.image_id, clones), ("im-1", [True]))
        self.write("edited.py", "a Python edit\n")
        self.assertEqual(image_for(), first)
        self.write("lib.cpp", "int b;\n")
        self.write(".ci/docker/README.md", "a new CI image tag\n")  # same lineage
        self.assertEqual((image_for().image_id, clones), ("im-2", [True, False]))
        self.assertEqual(image_for(days_later=1).image_id, "im-2")
        self.assertEqual(image_for(days_later=4.1).image_id, "im-3")  # renewed
        self.assertEqual(image_for(days_later=10).image_id, "im-4")  # expired
        self.assertEqual(image_for(days_later=10, python="3.11").image_id, "im-4")
        self.assertEqual(image_for(days_later=10, python="3.13").image_id, "im-5")
        self.assertEqual(clones, [True, False, False, True, True])

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_sync_files(self) -> None:
        runner = self.tmp / "runner"
        runner.mkdir()
        sync.archive(self.base, ["."], self.tmp / "base.tar")
        subprocess.run(["tar", "-xf", self.tmp / "base.tar"], cwd=runner, check=True)
        self.edit()
        commit = sync.snapshot_tree()
        run.sync_files(LocalRunner(runner), self.base, commit, self.tmp)
        self.assertEqual(files(runner), files(self.repo))

    def test_times(self) -> None:
        files = {"test/test_a.py": 3, "test/it's.py": 2}
        with self.assertLogs("ramjet", "WARNING") as logs:  # no wrapper to run
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})
        self.assertIn("no test times from CI:\n", logs.output[0])
        failing = "#!/bin/sh\necho gh: not logged in >&2\necho gcx.error\nexit 1\n"
        self.write(times.WRAPPER, failing)
        (self.repo / times.WRAPPER).chmod(0o755)
        with self.assertLogs("ramjet", "WARNING") as logs:
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})
        message = "no test times from CI:\ngh: not logged in\ngcx.error"
        self.assertEqual(logs.records[0].getMessage(), message)
        one, two = "test/test_a.py::A::test_one", "test/test_a.py::test_two"
        self.answer([[one, 1.5, 2.0], [two, None, 0.25]])
        fetched = times.fetch(files, "NVIDIA L4", 1)
        self.assertEqual(fetched, {one: (1.5, 2.0), two: (None, 0.25)})
        query = (self.tmp / "queries").read_text()
        self.assertIn("e.device_name = 'NVIDIA L4' AND e.device_count = 1", query)
        self.assertIn("t.file IN ('test/test_a.py', 'test/it\\'s.py')", query)
        with mock.patch.object(times, "BATCH", 4):  # a query for each file, then
            times.fetch(files, "", 0)
        queries = (self.tmp / "queries").read_text().split("SELECT")[2:]
        self.assertEqual(len(queries), 2)
        self.assertIn("e.device_name = '' AND e.device_count = 0", queries[0])
        (self.tmp / "rows.json").write_text('{"rows": null}')  # CI ran none of them
        with self.assertNoLogs("ramjet", "WARNING"):
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_list_tests(self) -> None:
        container = self.tmp / "container"  # its /pytorch, at the build image's commit
        container.mkdir()
        sync.archive(self.base, ["."], self.tmp / "base.tar")
        subprocess.run(["tar", "-xf", self.tmp / "base.tar"], cwd=container, check=True)
        self.edit()
        self.write("test/test_file.py", TEST_FILE)
        self.write("test/test_broken.py", "raise ImportError('no such module')\n")
        launched = f"--use-pytest --collect-only -p {worker.TCI} -k second -v"
        second = "test/test_file.py::TestFile::test_second"
        nodeid = f"{second}[{launched} --verbosity=-1 no PYTHONPATH]"
        self.answer([[nodeid, 2.0, 3.0]])
        commit = sync.snapshot_tree()

        def local_map(names: list[str], kwargs: Any, return_exceptions: bool) -> Any:
            started = worker.Worker()  # this process stands in for the container
            started.sync()
            lost = {"test_lost": ConnectionError("container lost")}
            return [lost.get(n) or started.collect(n, **kwargs) for n in names]

        modal = mock.MagicMock()  # stands in for Modal
        image = modal.Image.from_id.return_value.run_commands.return_value

        def add_local_file(local: str, remote: str) -> Any:
            shutil.copyfile(local, remote)  # where the container finds it
            return image

        image.add_local_file.side_effect = add_local_file
        new_app = modal.App.return_value
        new_app.cls.return_value.return_value.return_value.collect.map = local_map

        def fetch(*args: Any) -> Any:
            new_app.run.return_value.__exit__.assert_called_once()  # containers stopped
            return mock.DEFAULT  # what times.fetch returns

        modal_sets = {"PYTHONPATH": "/pkg/:/root/", "MODAL_TASK_ID": "ta-1"}
        for patcher in [
            mock.patch.object(test, "modal", modal),
            mock.patch.object(times, "fetch", wraps=times.fetch, side_effect=fetch),
            mock.patch.object(worker, "ROOT", str(container)),
            mock.patch.object(worker, "CHANGES", str(self.tmp / "changes.tar")),
            mock.patch.object(worker, "DELETED", str(self.tmp / "deleted")),
            mock.patch.object(worker, "machine", return_value=L4),
            mock.patch.dict(os.environ, modal_sets),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

        build_image = lineage.Build("im-1", self.base, taken=0.0)
        sizes = (test.CORES_PER_GPU, test.MIB_PER_CORE)
        resources = run.resources("L4", None, None, *sizes)
        names = {
            "test_file": ["test/test_file.py::TestFile::test_second"],
            "test_broken": ["test/test_broken.py"],
            "test_lost": ["test/test_lost.py"],
        }
        args = (build_image, commit, resources, names, ["-k", "second", "-v"], self.tmp)
        with self.assertLogs(test.log, "DEBUG") as logs:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(test.list_tests(*args), 1)
        listed = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(listed, [{"nodeid": nodeid, "piece": 1}])
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(messages[0], "collecting 3 files on L4")
        took = r"^test/test_%s\.py took \d+\.\d s on container ta-1$"
        self.assertRegex(messages[1], took % "file")
        # Of the two tests tci left out, the one under the node id asked for
        needs = "@tci.needs(tci.gpus >= 2)"
        self.assertEqual(messages[2], f"left out {second}[two]: {needs}")
        self.assertRegex(messages[3], took % "broken")
        self.assertIn("test/test_broken.py failed:", messages[4])
        self.assertIn("ImportError: no such module", messages[4])
        self.assertIn("test/test_lost.py failed:", messages[5])
        self.assertIn("container lost", messages[5])
        left_out = ["left out 1 test that L4 can't run:", f"  1 with {needs}"]
        self.assertEqual(messages[6:9], ["listed 1 test from 1 file", *left_out])
        # CI took 2 s for the test on an L4, which is the machine it was asked about
        sources = "1 test with 1 x NVIDIA L4, 0 on other machines, 0 guessed"
        seconds = round(test.START_SECONDS + 2 * test.SLOWDOWN)
        planned = [
            f"CI's times: {sources}",
            f"piece 1, about {seconds} s: 1 test of 1 file",
            f"1 piece: 1 container-minute, the longest about {seconds} s",
        ]
        self.assertEqual(messages[9:], planned)
        query = (self.tmp / "queries").read_text()
        self.assertIn("e.device_name = 'NVIDIA L4' AND e.device_count = 1", query)
        self.assertIn("t.file IN ('test/test_file.py')", query)
        self.assertEqual(files(container), files(self.repo))  # synced first
        options = {
            "image": image,
            "scaledown_window": test.SCALEDOWN_SECONDS,
            "gpu": "L4",
            "cpu": 4,
            "memory": 16384,
        }
        new_app.cls.assert_called_once_with(**options)
        modal.concurrent.assert_called_once_with(max_inputs=4)  # a file per core
        modal.concurrent.return_value.assert_called_once_with(worker.Worker)

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    @unittest.skipUnless(shutil.which("setsid"), "needs util-linux's setsid")
    def test_execute(self) -> None:
        (self.repo / "sub").mkdir()
        command = ["sh", "-c", 'printf %s "$1" > out; exit 3', "sh", "a b"]
        with mock.patch.object(run, "PID_FILE", str(self.tmp / "pid")):
            self.assertEqual(run.execute(LocalRunner(self.repo), command, "sub"), 3)
        self.assertEqual((self.repo / "sub/out").read_text(), "a b")  # as given
        self.assertRegex((self.tmp / "pid").read_text(), r"^\d+\n$")


class TestLineage(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_build_config(self) -> None:
        environ = {
            "BUILD_TEST": "0",
            "CMAKE_PREFIX_PATH": "/opt/conda",
            "CUDA_PATH": "/usr/local/cuda",
            "DEBUG": "1",
            "MAX_JOBS": "8",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "USE_CUDA": "1",
        }
        names = ["BUILD_TEST", "DEBUG", "TORCH_CUDA_ARCH_LIST", "USE_CUDA"]
        self.assertEqual(lineage.build_config(environ), {n: environ[n] for n in names})

    def test_key(self) -> None:
        config = {"USE_CUDA": "1", "DEBUG": "1"}
        key = lineage.key("ci-image:a", config)
        self.assertEqual(key, lineage.key("ci-image:a", dict(reversed(config.items()))))
        self.assertNotEqual(key, lineage.key("ci-image:b", config))
        self.assertNotEqual(key, lineage.key("ci-image:a", {"USE_CUDA": "1"}))

    def test_usable_and_due(self) -> None:
        build = lineage.Build("im-1", "abc", taken=0.0)
        self.assertIsNone(lineage.usable(None, 0.0))
        self.assertIs(lineage.usable(build, lineage.EXPIRES_AFTER - 1), build)
        self.assertIsNone(lineage.usable(build, lineage.EXPIRES_AFTER))
        self.assertFalse(lineage.due(build, lineage.RENEW_AFTER))
        self.assertTrue(lineage.due(build, lineage.RENEW_AFTER + 1))

    def test_read_by_build(self) -> None:
        read = frozenset({"aten/src/ATen/native/Linear.cpp", "torchgen/gen.py"})
        cases = [
            ("torch/nn/modules/linear.py", "M", False),
            ("test/test_new.py", "A", False),
            ("torch/csrc/README.md", "M", False),
            ("aten/src/ATen/native/Linear.cpp", "M", True),
            ("torchgen/gen.py", "D", True),
            ("torch/csrc/new.cpp", "A", True),
            ("caffe2/CMakeLists.txt", "M", True),
            ("cmake/Modules/FindFoo.cmake", "M", True),
            ("setup.py", "M", True),
            (".ci/docker/ci_commit_pins/triton.txt", "M", True),  # a Triton pin bump
            ("third_party/fbgemm", "M", True),  # a submodule bump
            ("torch/_native/ops.py", "M", True),
        ]
        for path, status, builds in cases:
            with self.subTest(path=path, status=status):
                found = lineage.read_by_build({path: status}, read)
                self.assertEqual(found, [path] if builds else [])
        changes = {
            "setup.py": "M",
            "torch/nn/functional.py": "M",
            "aten/src/ATen/native/Linear.cpp": "M",
        }
        found = lineage.read_by_build(changes, read)
        self.assertEqual(found, ["setup.py", "aten/src/ATen/native/Linear.cpp"])

    def test_load_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch, "lineage.json")
            self.assertEqual(lineage.load(path), {})
            builds = {"key": lineage.Build("im-1", "abc", 1.5)}
            lineage.save(path, builds)
            self.assertEqual(lineage.load(path), builds)


@skipIfWindows(msg="ramjet runs on Linux and macOS")
@unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"), "needs CMake")
class TestInputs(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_build_inputs(self) -> None:
        sources = {
            "CMakeLists.txt": CMAKE_LISTS,
            "demo.c": '#include "demo.h"\nint demo(void) { return DEMO; }\n',
            "demo.h": "#define DEMO 1\n",
        }
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            for name, text in sources.items():
                (root / name).write_text(text)
            build = root / "build"
            cmake = ["cmake", "-G", "Ninja", "-S", root, "-B", build]
            subprocess.run(cmake, check=True, capture_output=True)
            subprocess.run(["ninja", "-C", build], check=True, capture_output=True)
            self.assertEqual(inputs.build_inputs(str(root)), sorted(sources))


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestRun(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_resources(self) -> None:
        self.assertEqual(run.resources(None, None, None), (None, 8, 65536))
        self.assertEqual(run.resources("H100:2", None, None), ("H100:2", 16, 131072))
        self.assertEqual(run.resources("B200:8", None, None), ("B200:8", 64, 344064))
        self.assertEqual(run.resources(None, 32, None), (None, 32, 262144))
        self.assertEqual(run.resources("B200:8", 2, None), ("B200:8", 2, 16384))
        self.assertEqual(run.resources("H100", 64, 8192), ("H100", 64, 8192))
        # test's containers are smaller
        sizes = (test.CORES_PER_GPU, test.MIB_PER_CORE)
        self.assertEqual(run.resources(None, None, None, *sizes), (None, 4, 16384))
        two = run.resources("H100:2", None, None, *sizes)
        self.assertEqual(two, ("H100:2", 8, 32768))
        self.assertEqual(run.resources("L4", 8, None, *sizes), ("L4", 8, 32768))

    def test_hourly_rate(self) -> None:
        def rate(*request: Any) -> Decimal:  # from --gpu, --cpu and --memory
            return run.hourly_rate(run.resources(*request), RATES)

        core, gib = RATES["cpu_hour_cost_sandbox"], RATES["mem_gib_hour_cost_sandbox"]
        host = 8 * core + 64 * gib  # 8 cores and 64 GiB
        self.assertEqual(rate(None, None, None), host)
        self.assertEqual(rate(None, 32, None), 32 * core + 256 * gib)
        self.assertEqual(rate(None, 8, 8192), 8 * core + 8 * gib)
        gpus = {
            "H100": "h100",
            "H100!": "h100",
            "A10": "a10g",
            "A100": "a100_40gb",
            "A100-80GB": "a100_80gb",
            "RTX-PRO-6000": "rtx6000",
            "B200+": "b200",
        }
        for gpu, name in gpus.items():
            with self.subTest(gpu=gpu):
                gpu_rate = RATES[f"gpu_hour_cost_{name}"]
                self.assertEqual(rate(gpu, None, None), host + gpu_rate)
        no_h100 = {k: v for k, v in RATES.items() if k != "gpu_hour_cost_h100"}
        with exits(self, "no GPU named h100"):
            run.hourly_rate(run.resources("H100", None, None), no_h100)

    def test_idle_timeout(self) -> None:
        def idle(gpu: str | None) -> int:
            return run.idle_timeout(run.resources(gpu, None, None), RATES)

        self.assertEqual(idle(None), run.IDLE_MAX_SECONDS)
        self.assertEqual(idle("H100"), run.IDLE_MAX_SECONDS)
        # 8 B200s with 64 cores and 336 GiB bill $67.15 an hour: $2.50 is 134 s
        self.assertEqual(idle("B200:8"), 134)

    def test_show(self) -> None:
        h100, cpu = run.resources("H100:8", None, None), run.resources(None, None, None)
        with self.assertLogs(run.log, "INFO") as logs:
            run.show("sb-new", False, h100, Decimal("48.75"))
            run.show("sb-warm", True, cpu, Decimal("2.672"))
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [
                "sandbox sb-new (new): \033[31m$48.75/h\033[0m • H100 × 8 • 64 cores • 336 GiB",
                "sandbox sb-warm (warm): \033[31m$2.67/h\033[0m • no GPU • 8 cores • 64 GiB",
            ],
        )

    def test_runs_on(self) -> None:
        cuda13 = "7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        cuda12 = "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
        for name, gpu in run.GPUS.items():
            with self.subTest(gpu=name):
                blackwell = gpu.capability >= (10, 0)
                self.assertTrue(run.runs_on(gpu.capability, cuda13))
                self.assertEqual(run.runs_on(gpu.capability, cuda12), not blackwell)
        self.assertFalse(run.runs_on((10, 0), "9.0"))
        self.assertTrue(run.runs_on((10, 0), "9.0+PTX"))  # the driver compiles PTX
        self.assertTrue(run.runs_on((8, 9), "7.5 8.6"))  # CI separates with spaces
        self.assertTrue(run.runs_on((10, 0), "Hopper"))  # not judged
        with exits(self, "can't run on B200:8"):
            run.check_archs("B200:8", cuda12)
        run.check_archs(None, cuda12)


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestBuild(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_ci_image_name(self) -> None:
        names = [
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3.11-gcc11",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11-inductor-benchmarks",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3.12-gcc11",
            "pytorch-linux-jammy-cuda13.4-cudnn9-py3.11-clang20",
        ]
        found = build.CI_IMAGE_NAME.findall(",\n".join(names) + ",")
        self.assertEqual(found, [(names[0], "12.6"), (names[1], "13.2")])

    def test_ci_python(self) -> None:
        script = (
            "  pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11)\n"
            "    CUDA_VERSION=13.2.2\n"
            "    ANACONDA_PYTHON_VERSION=3.11\n"
            "    ;;\n"
            "  pytorch-linux-jammy-cuda13.2-cudnn9-py3.12-gcc11)\n"
            "    ANACONDA_PYTHON_VERSION=3.12\n"
            "    ;;\n"
        )
        name = "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11"
        self.assertEqual(build.ci_python(script, name), "3.11")
        with exits(self, "no Python"):
            build.ci_python(script, name.replace("13.2", "12.8"))


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestCli(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_parse_args(self) -> None:
        def parse(*argv: str) -> tuple[object, ...]:
            options, command = cli.parse_args(list(argv))
            return (*vars(options).values(), command)  # in the order they're added

        cudas, stable = list(build.ci_images()), build.stable_cuda()
        pythons = build.python_versions()
        self.assertIn(stable, cudas)
        # Every option but --gpu, and no subcommand
        defaults = (None, None, stable, None, "info", None)
        self.assertEqual(parse("--", "ls"), (None, *defaults, ["ls"]))
        command = ["pytest", "test/test_nn.py", "-k", "linear or conv"]
        argv = ["--gpu", "H100", "--", *command]
        self.assertEqual(parse(*argv), ("H100", *defaults, command))
        cuda, python = f"--build-cuda={cudas[-1]}", f"--python-version={pythons[-1]}"
        options = ["--cpu=16", "--memory=131072", cuda, python, "--log-level=debug"]
        argv = ["--gpu=b200+:8", *options, "--", "nvidia-smi"]  # in any case
        expected = ("B200+:8", 16, 131072, cudas[-1], pythons[-1], "debug", None)
        self.assertEqual(parse(*argv), (*expected, argv[-1:]))
        argv = ["--", "a", "--", "b"]  # later --s are the command's
        self.assertEqual(parse(*argv), (None, *defaults, ["a", "--", "b"]))
        # test: its own options and the paths, then pytest's arguments after --
        paths = ["test/dynamo", "test/test_nn.py::TestNN"]
        listing = ("test", True, paths)
        argv = ["--gpu", "L4", "test", "--list", *paths]
        self.assertEqual(parse(*argv), ("L4", *defaults[:-1], *listing, []))
        argv = ["test", *paths, "--list", "--", "-k", "linear", "--", "-x"]
        pytest_args = ["-k", "linear", "--", "-x"]
        self.assertEqual(parse(*argv), (None, *defaults[:-1], *listing, pytest_args))
        bad = [
            [],  # no command
            ["--gpu", "L4", "--"],
            ["ls"],  # the command follows a --
            ["--gpu", "--", "ls"],
            ["--gpu", "Z100", "--", "ls"],
            ["--gpu", "A10:5", "--", "ls"],  # A10s go up to 4
            ["--cpu=3", "--", "ls"],
            ["--memory=1000", "--", "ls"],
            ["--build-cuda=9.9", "--", "ls"],
            ["--python-version=3.9", "--", "ls"],
            ["--log-level=dev", "--", "ls"],
            ["test", "--list"],  # no paths
            ["test", "test/dynamo"],  # only --list is built
            ["test", "--gpu", "L4", "--list", "test/dynamo"],  # --gpu goes before test
            ["test/dynamo"],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*argv)


@skipIfWindows(msg="ramjet runs on Linux and macOS")
@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestTest(TestCase):
    hw_classification = HardwareClassification.GENERIC
    THIS = "tools/ramjet/test_ramjet"  # this file, as run_test.py names it

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(REPO_ROOT)
        common.repo_root.cache_clear()
        self.addCleanup(common.repo_root.cache_clear)

    def test_names(self) -> None:
        file_id = f"test/{self.THIS}.py"
        self.assertEqual(test.names([file_id]), {self.THIS: [file_id]})
        self.assertEqual(test.names(["test/tools/ramjet"]), {self.THIS: [file_id]})
        node_id = f"{file_id}::TestCli::test_parse_args"
        both = [node_id, file_id]
        self.assertEqual(test.names(both), {self.THIS: both})
        os.chdir("test/tools")  # paths are from where you are
        self.assertEqual(test.names(["ramjet/test_ramjet.py"]), {self.THIS: [file_id]})
        os.chdir(REPO_ROOT)
        everything = test.names(["test"])
        self.assertIn(self.THIS, everything)
        self.assertIn("test_torch", everything)
        # run_test.py also knows names that are not test files, such as doctests
        no_file = [n for n in everything if not Path(f"test/{n}.py").is_file()]
        self.assertEqual(no_file, [])
        bad = [
            "test/fx",  # run by test_fx.py, not on their own
            "test/fx/test_common_passes.py",
            "test/missing.py",
            "test/tools/ramjet::TestCli",  # a node id names a file
            "tools/ramjet",  # not under test/
        ]
        for path in bad:
            with self.subTest(path=path), exits(self, f"-- pytest {path}"):
                test.names([path])

    def test_collect(self) -> None:
        with tempfile.NamedTemporaryFile() as nothing:  # changed or deleted
            here = {"ROOT": str(REPO_ROOT), "CHANGES": nothing.name}
            with mock.patch.multiple(worker, DELETED=nothing.name, **here):
                started = worker.Worker()
                started.sync()
                # This file, launched as run_test.py launches it
                collected = started.collect(self.THIS, ["-k", "test_names", "-v"])
                missing = started.collect("tools/ramjet/missing", [])
        ids = [f"test/{self.THIS}.py::TestTest::test_names"]
        self.assertEqual((collected["ids"], collected["error"]), (ids, ""))
        self.assertEqual(collected["left_out"], {})  # no test here has a constraint
        self.assertGreater(collected["seconds"], 0)
        read = json.loads(json.dumps(dataclasses.asdict(machine.capture())))
        self.assertEqual(collected["machine"], read)
        self.assertEqual(missing["ids"], [])
        self.assertIn("missing.py", missing["error"])

    def test_timed(self) -> None:
        kept = {
            "test_a": ["test/test_a.py::A::test_here", "test/test_a.py::A::test_other"],
            "test_b": [f"test/test_b.py::B::test_{n}" for n in range(4)],
            "test_new": ["test/test_new.py::New::test_new"],
        }
        ci = {
            "test/test_a.py::A::test_here": (4.0, 1.0),
            "test/test_a.py::A::test_other": (None, 2.0),
            "test/test_b.py::B::test_0": (1.0, 5.0),
            "test/test_b.py::B::test_2": (3.0, 5.0),
        }
        with (
            mock.patch.object(times, "fetch", return_value=ci) as fetch,
            self.assertLogs(test.log, "DEBUG") as logs,
        ):
            timed = test.timed(kept, L4)
        fetch.assert_called_once_with(
            {"test/test_a.py": 2, "test/test_b.py": 4, "test/test_new.py": 1},
            "NVIDIA L4",
            1,
        )
        # CI's seconds: here, else anywhere, else its file's median, else a guess
        expected = {
            "test_a": [4.0, 2.0],
            "test_b": [1.0, 2.0, 3.0, 2.0],
            "test_new": [test.GUESS_SECONDS],
        }
        seconds = {name: [s / test.SLOWDOWN for _, s in timed[name]] for name in timed}
        self.assertEqual(seconds, expected)
        ids = {name: [nodeid for nodeid, _ in tests] for name, tests in timed.items()}
        self.assertEqual(ids, kept)
        sources = "3 tests with 1 x NVIDIA L4, 1 on other machines, 3 guessed"
        guessed = "guessed the time of 3 tests that CI has none for"
        logged = [record.getMessage() for record in logs.records]
        self.assertEqual(logged, [f"CI's times: {sources}", guessed])
        self.assertEqual(logs.records[1].levelname, "INFO")

    def test_progress(self) -> None:
        def wait_for(message: str) -> None:
            deadline = time.monotonic() + 60
            while message not in [record.getMessage() for record in logs.records]:
                self.assertLess(time.monotonic(), deadline, f"never logged: {message}")
                time.sleep(0.01)

        often = mock.patch.object(test, "PROGRESS_SECONDS", 0.01)
        collected = {"container": "ta-1"}
        with often, self.assertLogs(test.log, "INFO") as logs:
            with test.progress(5) as back:
                wait_for("  no file collected yet: waiting for containers")
                back += [collected, ConnectionError("container lost"), collected]
                wait_for("  3 of 5 files collected by 1 container")
                back.append({"container": "ta-2"})
                wait_for("  4 of 5 files collected by 2 containers")
            logged = len(logs.records)
            time.sleep(0.05)
            self.assertEqual(len(logs.records), logged)  # it stopped with the block

    def test_shard(self) -> None:
        start = test.START_SECONDS
        timed = {
            "big": [(f"big::{n}", 40.0) for n in range(10)],  # three fit in a piece
            "small": [("small::0", 10.0), ("small::1", 10.0)],
            "medium": [("medium::0", 100.0)],
            "huge": [("huge::0", 400.0)],  # longer than a piece
        }
        pieces = test.shard(timed)
        planned = [(round(p.seconds - start * len(p.tests)), p.tests) for p in pieces]
        expected = [
            (400, {"huge": ["huge::0"]}),
            # small goes to the first piece with room for it, of the longest first
            (120, {"medium": ["medium::0"], "small": ["small::0", "small::1"]}),
            (120, {"big": ["big::0", "big::1", "big::2"]}),
            (120, {"big": ["big::3", "big::4", "big::5"]}),
            (120, {"big": ["big::6", "big::7", "big::8"]}),
            (40, {"big": ["big::9"]}),
        ]
        self.assertEqual(planned, expected)


if __name__ == "__main__":
    run_tests()
