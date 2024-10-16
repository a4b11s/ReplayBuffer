import unittest
from unittest.mock import MagicMock, patch
from replaybuffer.replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    @patch('replaybuffer.replay_buffer.DiskManager')
    @patch('replaybuffer.replay_buffer.Prefetcher')
    @patch('replaybuffer.replay_buffer.BackgroundSaver')
    @patch('filelock.FileLock')
    def setUp(self, MockFileLock, MockBackgroundSaver, MockPrefetcher, MockDiskManager):
        self.max_size = 100
        self.h5_path = '/tmp/test.h5'
        self.image_shape = (84, 84, 3)
        self.device = 'cpu'
        self.batch_size = 32

        self.mock_lock = MockFileLock.return_value
        self.mock_disk_manager = MockDiskManager.return_value
        self.mock_prefetcher = MockPrefetcher.return_value
        self.mock_background_saver = MockBackgroundSaver.return_value

        self.replay_buffer = ReplayBuffer(self.max_size, self.h5_path, self.image_shape, self.device, self.batch_size)

    def test_init_h5_file(self):
        shapes = {
            "state": self.image_shape,
            "action": (1,),
            "reward": (1,),
            "next_state": self.image_shape,
            "done": (1,),
        }
        self.mock_disk_manager._init_h5_file.assert_called_once_with(shapes)

    def test_start_subprocesses(self):
        self.mock_prefetcher.start.assert_called_once()
        self.mock_background_saver.start.assert_called_once()

    def test_add(self):
        state = [[0] * 84] * 84
        action = 1
        reward = 1.0
        next_state = [[0] * 84] * 84
        done = False

        self.replay_buffer.add(state, action, reward, next_state, done)
        self.mock_background_saver.save.assert_called_once_with(
            {
                "state": state,
                "action": [action],
                "reward": [reward],
                "next_state": next_state,
                "done": done,
            }
        )

    def test_sample(self):
        self.replay_buffer.sample()
        self.mock_prefetcher.sample.assert_called_once()

    def test_del(self):
        del self.replay_buffer
        self.mock_background_saver.stop.assert_called_once()
        self.mock_prefetcher.stop.assert_called_once()
        self.mock_disk_manager.lock.release.assert_called_once()
        self.mock_lock.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()