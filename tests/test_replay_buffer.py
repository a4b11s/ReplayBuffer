import tempfile
import unittest
import os
import numpy as np
import torch
from replaybuffer.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.max_size = 100
        self.h5_path = "/tmp/test_replay_buffer.h5"
        self.image_shape = (3, 64, 64)
        self.device = torch.device("cpu")
        self.batch_size = 10
        
        try:
            os.remove(self.h5_path)
        except FileNotFoundError:
            pass
        
        self.replay_buffer = ReplayBuffer(
            self.max_size, self.h5_path, self.image_shape, self.device, self.batch_size
        )

    def _delete_h5(self):
        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)

    def tearDown(self):
        self._delete_h5()

    def test_add_and_sample(self):
        state = np.random.rand(*self.image_shape).astype(np.float32)
        action = np.random.randint(0, 10)
        reward = np.random.rand()
        next_state = np.random.rand(*self.image_shape).astype(np.float32)
        done = False

        for _ in range(self.batch_size):
            self.replay_buffer.add(state, action, reward, next_state, done)

        # Wait for the samples to be saved
        while self.replay_buffer.save_queue.empty():
            pass

        # Wait for the sample to be available
        while self.replay_buffer.prefetch_batches.empty():
            pass

        sample = self.replay_buffer.sample()
        self.assertEqual(len(sample), self.batch_size)
        (
            sampled_state,
            sampled_action,
            sampled_reward,
            sampled_next_state,
            sampled_done,
        ) = sample[0]

        np.testing.assert_array_almost_equal(sampled_state.cpu().numpy(), state)
        self.assertEqual(sampled_action.item(), action)
        self.assertAlmostEqual(sampled_reward.item(), reward)
        np.testing.assert_array_almost_equal(
            sampled_next_state.cpu().numpy(), next_state
        )
        self.assertEqual(sampled_done.item(), done)

    def test_save_and_load(self):
        for _ in range(self.batch_size):
            state = np.random.rand(*self.image_shape).astype(np.float32)
            action = np.random.randint(0, 10)
            reward = np.random.rand()
            next_state = np.random.rand(*self.image_shape).astype(np.float32)
            done = False
            self.replay_buffer.add(state, action, reward, next_state, done)

        # Wait for the samples to be saved
        while not self.replay_buffer.save_queue.empty():
            pass

        loaded_batch = self.replay_buffer._load_batch_from_disk()
        self.assertEqual(len(loaded_batch), self.batch_size)


if __name__ == "__main__":
    unittest.main()
