"""
tools/launcher/pacbot-train, run against a temporary state directory and a fake checkout whose
"python" records how it was called: each algorithm's script and environment, where checkpoints
go, starting from another run's checkpoint, resuming from a run's own latest checkpoint, and
job records written before --algorithm existed (a teammate's queued run during an upgrade).

Needs bash, flock and setsid (Linux); skipped elsewhere.

Run from src/: python -m unittest discover ../tests
"""

import os
import shutil
import subprocess
import tempfile
import time
import textwrap
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "tools" / "launcher" / "pacbot-train"
MEMWATCH = LAUNCHER.with_name("pacbot-memwatch")
HAVE_TOOLS = all(shutil.which(t) for t in ("bash", "flock", "setsid", "pgrep")) and os.name == "posix"

FAKE_PYTHON = textwrap.dedent("""\
    #!/bin/bash
    # Records its arguments and environment, trains for $FAKE_SLEEP seconds (default 0; SIGTERM
    # stops it, as it would a real run), then exits with $FAKE_EXIT (default 0)
    {{ echo "python=$0"; echo "cwd=$PWD"; echo "WANDB_NAME=$WANDB_NAME"; printf 'arg=%s\\n' "$@"; echo; }} >> {calls}
    trap 'exit 143' TERM
    sleep ${{FAKE_SLEEP:-0}} & wait
    exit ${{FAKE_EXIT:-0}}
    """)


@unittest.skipUnless(HAVE_TOOLS, "needs bash, flock and setsid")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.state = self.tmp / "state"
        (self.state / "jobs").mkdir(parents=True)
        self.calls = self.tmp / "calls"
        self.top = self.tmp / "checkout"
        for script in ("dqn", "alphazero", "ppo", "lightzero"):
            path = self.top / "src" / "algorithms" / f"train_{script}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for venv in (".venv", "lightzero/.venv"):
            python = self.top / venv / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text(FAKE_PYTHON.format(calls=self.calls))
            python.chmod(0o755)
        subprocess.run(["git", "init", "-q", str(self.top)], check=True)

    def launcher(self, *args, exit_code=0, cwd=None, owner="the-owner", sleep=0, max_runs=99):
        env = {**os.environ, "PACBOT_TRAIN_STATE": str(self.state), "PACBOT_TRAIN_MAX": str(max_runs),
               "PACBOT_TRAIN_POLL": "1", "PACBOT_TRAIN_MATCH": "no-such-training-process",
               "PACBOT_TRAIN_OWNER": owner, "FAKE_EXIT": str(exit_code), "FAKE_SLEEP": str(sleep)}
        env.pop("PACBOT_TRAIN_PYTHON", None)
        return subprocess.run(["bash", str(LAUNCHER), *args], cwd=cwd or self.top, env=env,
                              capture_output=True, text=True, timeout=60)

    def last_call(self):
        call = self.calls.read_text().strip().split("\n\n")[-1].splitlines()
        fields = dict(line.split("=", 1) for line in call if not line.startswith("arg="))
        fields["args"] = [line[4:] for line in call if line.startswith("arg=")]
        return fields

    def test_dqn_by_default_as_before(self):
        result = self.launcher("run", "d1", "--", "--single-model", "--eval_freq", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.last_call()
        self.assertEqual(call["python"], str(self.top / ".venv/bin/python"))
        self.assertEqual(call["cwd"], str(self.top / "src"))
        self.assertEqual(call["WANDB_NAME"], "d1")
        ckpt = self.top / "src/checkpoints/d1"
        self.assertEqual(call["args"], ["-u", "-m", "algorithms.train_dqn", "--checkpoint-dir", str(ckpt),
                                        "--single-model", "--eval_freq", "5"])
        self.assertTrue((ckpt / "done").exists())
        listing = self.launcher("list").stdout
        self.assertRegex(listing, r"d1\s+dqn\s+done")

    def test_dqn_resumes_from_its_latest_checkpoint(self):
        self.assertNotEqual(self.launcher("run", "d2", exit_code=1).returncode, 0)  # it failed
        ckpt = self.top / "src/checkpoints/d2"
        (ckpt / "q_net-latest.ckpt.pt").write_text("latest")
        self.assertEqual(self.launcher("run", "d2").returncode, 0)
        args = self.last_call()["args"]
        self.assertEqual(args[args.index("--finetune") + 1], str(ckpt / "resume-from.ckpt.pt"))
        self.assertEqual((ckpt / "resume-from.ckpt.pt").read_text(), "latest")

    def test_alphazero_from_another_runs_checkpoint_in_another_folder(self):
        root = self.tmp / "shared"
        (root / "old").mkdir(parents=True)
        (root / "old" / "model-latest.ckpt.pt").write_text("old run")
        result = self.launcher("run", "--algorithm", "alphazero", "--checkpoints", str(root),
                               "--from", str(root / "old/model-latest.ckpt.pt"), "a1", "--", "--num_moves", "9")
        self.assertEqual(result.returncode, 0, result.stderr)
        ckpt = root / "a1"
        call = self.last_call()
        self.assertEqual(call["args"], ["-u", "-m", "algorithms.train_alphazero", "--checkpoint-dir", str(ckpt),
                                        "--resume", str(ckpt / "resume-from.ckpt.pt"), "--num_moves", "9"])
        self.assertEqual((ckpt / "resume-from.ckpt.pt").read_text(), "old run")
        self.assertTrue((ckpt / "done").exists() and (ckpt / "train.log").exists())
        self.assertRegex(self.launcher("list").stdout, r"a1\s+alphazero\s+done")

    def test_its_own_latest_checkpoint_comes_before_from(self):
        root = self.tmp / "shared"
        (root / "p1").mkdir(parents=True)
        (root / "p1" / "model-latest.ckpt.pt").write_text("own")
        (root / "other.pt").write_text("other")
        self.launcher("run", "--algorithm", "ppo", "--checkpoints", str(root), "--from", str(root / "other.pt"), "p1")
        args = self.last_call()["args"]
        self.assertEqual(args[2], "algorithms.train_ppo")
        self.assertEqual((root / "p1" / "resume-from.ckpt.pt").read_text(), "own")

    def test_lightzero_uses_its_environment_and_newest_checkpoint(self):
        result = self.launcher("run", "--algorithm", "lightzero", "z1", "--", "--algorithm", "muzero", exit_code=1)
        call = self.last_call()
        self.assertEqual(call["python"], str(self.top / "lightzero/.venv/bin/python"))
        self.assertEqual(call["args"][2], "algorithms.train_lightzero")
        self.assertNotIn("--resume", call["args"])
        ckpt = self.top / "src/checkpoints/z1"
        for i, run in enumerate(["run-1", "run-2"]):
            (ckpt / run / "ckpt").mkdir(parents=True)
            path = ckpt / run / "ckpt" / f"iteration_{i}.pth.tar"
            path.write_text(run)
            os.utime(path, (1_000_000 + i, 1_000_000 + i))
        self.launcher("run", "--algorithm", "lightzero", "z1", "--", "--algorithm", "muzero")
        args = self.last_call()["args"]
        self.assertEqual(args[args.index("--resume") + 1], str(ckpt / "run-2/ckpt/iteration_1.pth.tar"))
        self.assertEqual(args[-2:], ["--algorithm", "muzero"])

    def test_records_from_before_algorithms_run_as_dqn(self):
        # A run queued by the previous version: no algorithm, ckpt or python fields
        ckpt = self.top / "src/checkpoints/old"
        ckpt.mkdir(parents=True)
        job = self.state / "jobs" / f"{os.environ.get('USER', 'u')}.old"
        job.write_text(f"user=x\nname=old\ndir={self.top / 'src'}\nseq=1\nstate=queued\n")
        (self.state / "jobs" / (job.name + ".args")).write_bytes(b"--single-model\0")
        env_python = self.top / ".venv/bin/python"
        result = subprocess.run(["bash", str(LAUNCHER), "__supervise", job.name], cwd=self.top,
                                env={**os.environ, "PACBOT_TRAIN_STATE": str(self.state), "PACBOT_TRAIN_MAX": "99",
                                     "PACBOT_TRAIN_POLL": "1", "PACBOT_TRAIN_MATCH": "no-such-training-process",
                                     "PACBOT_TRAIN_PYTHON": str(env_python)},
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.last_call()["args"], ["-u", "-m", "algorithms.train_dqn", "--checkpoint-dir", str(ckpt),
                                                    "--single-model"])
        self.assertTrue((ckpt / "done").exists())

    def teammate_run(self, name, state="done"):
        job = self.state / "jobs" / f"teammate.{name}"
        job.write_text(f"user=teammate\nname={name}\ndir={self.top / 'src'}\nseq=9\nstate={state}\n")
        (self.state / "jobs" / f"teammate.{name}.args").write_bytes(b"")
        return job

    def test_starting_again_by_name_keeps_its_settings(self):
        root = self.tmp / "shared"
        root.mkdir()
        self.launcher("run", "--algorithm", "ppo", "--checkpoints", str(root), "p2", "--",
                      "--model", "PVNetV2", "--entropy_coef", "0.05", exit_code=1)  # e.g. the machine restarted
        (root / "p2" / "model-latest.ckpt.pt").write_text("latest")
        self.assertEqual(self.launcher("run", "p2").returncode, 0)
        args = self.last_call()["args"]
        self.assertEqual(args[2], "algorithms.train_ppo")
        self.assertEqual(args[args.index("--checkpoint-dir") + 1], str(root / "p2"))
        self.assertEqual(args[-4:], ["--model", "PVNetV2", "--entropy_coef", "0.05"])
        self.assertRegex(self.launcher("list").stdout, r"p2\s+ppo\s+done")

    def test_new_arguments_replace_the_saved_ones(self):
        self.launcher("run", "d3", "--", "--eval_freq", "5", exit_code=1)
        self.launcher("run", "d3", "--", "--single-model")
        self.assertEqual(self.last_call()["args"][-1:], ["--single-model"])

    def state_of(self, name):
        job = self.state / "jobs" / f"{subprocess.run(['id', '-un'], capture_output=True, text=True).stdout.strip()}.{name}"
        return [l[6:] for l in job.read_text().splitlines() if l.startswith("state=")][-1]

    def wait_for(self, name, state, timeout=30):
        deadline = time.time() + timeout
        while self.state_of(name) != state:
            self.assertLess(time.time(), deadline, f"{name} stayed {self.state_of(name)}, not {state}")
            time.sleep(0.2)

    def test_a_memory_limit_pauses_the_newest_run_until_it_lifts(self):
        for name in ("old", "new"):
            self.assertEqual(self.launcher("start", name, sleep=120, max_runs=3).returncode, 0)
            self.wait_for(name, "running")
        (self.state / "memory-limit").write_text("1\n")
        self.wait_for("new", "paused")
        self.assertEqual(self.state_of("old"), "running")
        self.assertIn("lowered the limit", self.launcher("list", max_runs=3).stdout)
        (self.state / "memory-limit").unlink()
        self.wait_for("new", "running")
        for name in ("old", "new"):
            self.launcher("stop", name, max_runs=3)
            self.wait_for(name, "stopped")

    def test_forget_removes_your_own_finished_run(self):
        self.assertEqual(self.launcher("run", "mine").returncode, 0)
        result = self.launcher("forget", "mine")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("mine", self.launcher("list").stdout)
        self.assertTrue((self.top / "src/checkpoints/mine/done").exists())  # its checkpoints stay

    def test_only_the_owner_can_forget_someone_elses_run(self):
        job = self.teammate_run("theirs")
        result = self.launcher("forget", "theirs", "teammate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only the-owner can remove other people's runs", result.stderr)
        self.assertTrue(job.exists())

        me = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
        result = self.launcher("forget", "theirs", "teammate", owner=me)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(job.exists())
        self.assertFalse(job.with_name(job.name + ".args").exists())

    def test_nobody_can_forget_a_run_in_progress(self):
        me = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
        job = self.teammate_run("busy", state="running")
        with job.open("a") as f:
            f.write(f"pid={os.getpid()}\n")  # a live process
        result = self.launcher("forget", "busy", "teammate", owner=me)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stop it first", result.stderr)
        self.assertTrue(job.exists())

    def test_rejects_what_the_launcher_sets_and_bad_options(self):
        for args in (["run", "x", "--", "--resume", "c.pt"], ["run", "x", "--", "--finetune=c.pt"],
                     ["run", "x", "--", "--eval", "c.pt"], ["run", "--algorithm", "sarsa", "x"],
                     ["run", "--from", "missing.pt", "x"], ["run", "--checkpoints", "/no/such/dir", "x"],
                     ["run", "--bogus", "x"]):
            with self.subTest(args=args):
                result = self.launcher(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pacbot-train:", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TOOLS, "needs bash")
class MemwatchTests(unittest.TestCase):
    def setUp(self):
        self.state = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.state)
        self.training = self.state / "training"
        fake = self.state / "fake-launcher"
        fake.write_text(f'#!/bin/bash\necho "Training now: $(cat {self.training}) of at most 3"\n')
        fake.chmod(0o755)
        self.fake = fake

    def check(self, used, training, calm=0):
        self.training.write_text(str(training))
        env = {**os.environ, "PACBOT_TRAIN_STATE": str(self.state), "PACBOT_TRAIN_MAX": "3",
               "PACBOT_MEM_CMD": f"echo {used}", "PACBOT_MEM_LAUNCHER": str(self.fake),
               "PACBOT_MEM_ONCE": "1", "PACBOT_MEM_CALM": str(calm), "PACBOT_MEM_SETTLE": "0"}
        subprocess.run(["bash", str(MEMWATCH)], env=env, check=True, timeout=30)
        limit = self.state / "memory-limit"
        return int(limit.read_text()) if limit.exists() else None

    def test_nearly_full_memory_allows_one_run_fewer(self):
        self.assertEqual(self.check(used=98, training=3), 2)
        self.assertEqual(self.check(used=97, training=2), 1)

    def test_nothing_changes_between_the_thresholds_or_with_nothing_training(self):
        self.assertIsNone(self.check(used=95, training=3))
        self.assertIsNone(self.check(used=99, training=0))

    def test_calm_memory_allows_runs_back_one_at_a_time(self):
        (self.state / "memory-limit").write_text("1\n")
        self.assertEqual(self.check(used=50, training=1), 2)
        self.assertIsNone(self.check(used=50, training=2))  # back to PACBOT_TRAIN_MAX: no limit

    def test_it_waits_for_memory_to_stay_calm(self):
        (self.state / "memory-limit").write_text("1\n")
        self.assertEqual(self.check(used=50, training=1, calm=600), 1)
