import logging
import sys
import unittest
import numpy as np
from replaybuffer.replay_buffer import ReplayBuffer


class TestReplayBufferE2E(unittest.TestCase):
    def test_replay_buffer_e2e(self):
        # Create a replay buffer
        batch_size = 256
        replay_buffer = ReplayBuffer(
            1000, "/tmp/test.h5", (84, 84, 3), "cpu", batch_size, 1000
        )

        # Data
        state = np.random.rand(84, 84, 3)
        action = [1]
        reward = [1.0]
        next_state = np.random.rand(84, 84, 3)
        done = [False]

        # Add data to the replay buffer
        for _ in range(600):
            replay_buffer.add(state, action, reward, next_state, done)

        for _ in range(32):
            result = replay_buffer.sample()
            self.assertEqual(result["state"].shape, (batch_size, 84, 84, 3))
            self.assertEqual(result["action"].shape, (batch_size, 1))
            self.assertEqual(result["reward"].shape, (batch_size, 1))
            self.assertEqual(result["next_state"].shape, (batch_size, 84, 84, 3))


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("BackgroundSaver").setLevel(logging.DEBUG)
    unittest.main()
