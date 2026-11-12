"""
Tests for parallel_training's actor processes and background evaluation workers, using the
real environment. Run from the repository root with PYTHONPATH=src.
"""

import time
import unittest

import torch

import models
import parallel_training
from utils import NUM_ACTIONS, OBS_SHAPE

CPU = torch.device("cpu")
STAY = 0


def favor(q_net: torch.nn.Module, stay_bias: float) -> None:
    """Make the network's Q values constant, with Stay offset by stay_bias."""
    with torch.no_grad():
        for p in q_net.parameters():
            p.zero_()
        q_net.advantage_head[0].bias[STAY] = stay_bias


class ActorReplayBufferTests(unittest.TestCase):
    device = CPU

    def make_buffer(self, q_net, num_envs=6, num_actors=2, steps=4, maxlen=10_000, ahead=None):
        return parallel_training.ActorReplayBuffer(
            maxlen=maxlen, q_net=q_net.to(self.device), model_name="QNetV2", epsilon=0.0,
            num_parallel_envs=num_envs, num_actors=num_actors, max_steps=steps,
            device=self.device, threads_per_actor=1, ahead_steps=ahead,
        )

    def collect_with_new_weights(self, buffer, steps=4, ahead=0):
        """Publish, drain what was played ahead with older weights, then collect."""
        buffer.start_collection(steps, 0.0)
        buffer.finish_collection()
        # On a GPU, weights reach the actors one publish later
        for _ in range(ahead // steps + 1):
            buffer.start_collection(steps, 0.0)
            buffer.finish_collection()
        first = buffer.storage.total
        buffer.start_collection(steps, 0.0)
        buffer.finish_collection()
        return [buffer.storage.transition(t) for t in range(first, buffer.storage.total)]

    def test_transitions_chain_per_environment(self):
        q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
        buffer = self.make_buffer(q_net, num_envs=5, num_actors=2)  # uneven split: 3 + 2
        try:
            for _ in range(3):
                buffer.start_collection(4, 0.5)
                buffer.finish_collection()
            storage = buffer.storage
            items = [storage.transition(t) for t in range(storage.total)]
            self.assertEqual(len(items), 3 * 4 * 5)
            self.assertEqual(storage.codec.mismatches(), 0)
            steps = [items[i:i + 5] for i in range(0, len(items), 5)]
            chained = 0
            for prev, cur in zip(steps, steps[1:]):
                for j in range(5):
                    self.assertEqual(len(cur[j]["next_action_mask"]), NUM_ACTIONS)
                    if prev[j]["next_obs"] is not None:
                        # This step's observation is the previous step's next observation
                        self.assertTrue(torch.equal(cur[j]["obs"], prev[j]["next_obs"]), (j,))
                        chained += 1
            self.assertGreater(chained, 40)
        finally:
            buffer.close()

    def test_actors_use_published_weights(self):
        q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
        buffer = self.make_buffer(q_net, ahead=4)
        try:
            favor(q_net, stay_bias=1e6)
            items = self.collect_with_new_weights(buffer, ahead=4)
            self.assertTrue(all(item["action"] == STAY for item in items))

            favor(q_net, stay_bias=-1e6)
            items = self.collect_with_new_weights(buffer, ahead=4)
            moves = [item["action"] != STAY for item in items]
            self.assertGreater(sum(moves) / len(moves), 0.9)
        finally:
            buffer.close()

    def test_fill_and_shutdown(self):
        buffer = self.make_buffer(models.QNetV2(OBS_SHAPE, NUM_ACTIONS), maxlen=100)
        procs = list(buffer._procs)
        try:
            buffer.fill()
            self.assertEqual(len(buffer), 100)
            self.assertEqual(buffer.sample(32).obs.shape[0], 32)
        finally:
            buffer.close()
        self.assertFalse(any(p.is_alive() for p in procs))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class CudaActorReplayBufferTests(ActorReplayBufferTests):
    """The same, with the buffer on the GPU (pinned copies, copy stream, background weights)."""

    device = torch.device("cuda")


@unittest.skipUnless(torch.backends.mps.is_available(), "needs an Apple Silicon GPU (MPS)")
class MpsActorReplayBufferTests(ActorReplayBufferTests):
    """The same, with the buffer on an Apple Silicon GPU."""

    device = torch.device("mps")


class BackgroundEvaluatorTests(unittest.TestCase):
    def test_evaluates_and_reports_payload(self):
        evaluator = parallel_training.BackgroundEvaluator(1, "QNetV2", env_version=2)
        try:
            q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
            self.assertTrue(evaluator.submit(7, q_net, lambda: "payload-7"))
            self.assertFalse(evaluator.submit(8, q_net, lambda: "payload-8"))  # worker busy
            self.assertEqual(evaluator.skipped, 1)
            deadline = time.monotonic() + 120
            finished = []
            while not finished and time.monotonic() < deadline:
                finished = evaluator.poll()
                time.sleep(0.05)
            self.assertEqual(len(finished), 1)
            iter_num, result, payload = finished[0]
            self.assertEqual((iter_num, payload), (7, "payload-7"))
            score, steps = result[0], result[1]
            self.assertGreaterEqual(score, 0)
            self.assertGreater(steps, 0)
            self.assertTrue(evaluator.has_capacity())
        finally:
            evaluator.close()

    def test_candidate_plays_every_speed_and_blocks_single_games(self):
        evaluator = parallel_training.BackgroundEvaluator(2, "QNetV2", env_version=2)
        try:
            q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
            evaluator.submit_candidate(100, q_net, [4, 8, 14], "snapshot-100", games_per_speed=2)
            self.assertEqual(evaluator.candidates_pending(), 1)
            # Its games fill the workers, so single games are skipped meanwhile
            self.assertFalse(evaluator.submit(101, q_net, lambda: None))
            deadline = time.monotonic() + 300
            finished = []
            while not finished and time.monotonic() < deadline:
                finished = evaluator.poll_candidates(timeout=0.1)
            self.assertEqual(len(finished), 1)
            iter_num, results, payload = finished[0]
            self.assertEqual((iter_num, payload), (100, "snapshot-100"))
            self.assertEqual(sorted(results), [4, 8, 14])
            self.assertTrue(all(len(games) == 2 for games in results.values()))
            self.assertTrue(all(r[1] > 0 for games in results.values() for r in games))
            self.assertEqual(evaluator.candidates_pending(), 0)
            self.assertTrue(evaluator.submit(102, q_net, lambda: None))
        finally:
            evaluator.close()

    def test_candidate_needs_a_game_per_speed(self):
        evaluator = parallel_training.BackgroundEvaluator(1, "QNetV2", env_version=2)
        try:
            with self.assertRaises(ValueError):
                evaluator.submit_candidate(1, models.QNetV2(OBS_SHAPE, NUM_ACTIONS), [8], None, games_per_speed=0)
        finally:
            evaluator.close()

if __name__ == "__main__":
    unittest.main()
