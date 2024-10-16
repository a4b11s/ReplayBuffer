import logging
import sys
import time
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

        start = time.time()
        # Add data to the replay buffer
        for _ in range(600):
            replay_buffer.add(state, action, reward, next_state, done)

        while replay_buffer.disk_manager.length < 600:
            time.sleep(0.1)
            
        print(f"Time to add 600 experiences: {time.time() - start}")
        start = time.time()
        for _ in range(100):
            result = replay_buffer.sample()
            print(f"{_}\r", end="")
        print(f"Time to sample 600 batches: {time.time() - start}")

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        # level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # logging.getLogger("Prefetcher").setLevel(logging.DEBUG)
    unittest.main()
