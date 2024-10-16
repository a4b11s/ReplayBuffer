import queue
import unittest
from unittest.mock import MagicMock, patch
import torch
from replaybuffer.prefetcher import Prefetcher
from replaybuffer.disk_manager import DiskManager


class TestPrefetcher(unittest.TestCase):
    def setUp(self):
        self.disk_manager = MagicMock(spec=DiskManager)
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.prefetcher = Prefetcher(self.disk_manager, self.device, self.batch_size)

    @patch("replaybuffer.prefetcher.Prefetcher._sample_batch_indicates")
    def test_process(self, mock_sample_batch_indicates):
        loaded_batch = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        self.disk_manager.length = 10
        mock_sample_batch_indicates.return_value = [0, 1, 2, 3]
        self.disk_manager.load_batch_from_disk.return_value = loaded_batch

        with patch.object(self.prefetcher.prefetch_batches, "put") as mock_put:
            self.prefetcher.run()
            self.prefetcher.stop()
            mock_put.assert_called_with(loaded_batch)

    def test_get_sample(self):
        sample_data = [[torch.tensor([1, 2, 3], device=self.device)]]
        self.prefetcher.prefetch_batches.put(sample_data)
        sample = self.prefetcher.get_sample()
        self.assertEqual(sample, sample_data)

    def test_sample_batch_indicates(self):
        self.disk_manager.length = 10
        indicates = self.prefetcher._sample_batch_indicates()
        self.assertEqual(len(indicates), self.batch_size)
        self.assertTrue(all(0 <= idx < self.disk_manager.length for idx in indicates))

    def test_run(self):
        self.prefetcher._process = MagicMock()
        self.prefetcher.run()
        self.assertTrue(self.prefetcher.running)

    def test_stop(self):
        self.prefetcher.thread = MagicMock()
        self.prefetcher.stop()
        self.assertFalse(self.prefetcher.running)
        self.prefetcher.thread.join.assert_called_once()


if __name__ == "__main__":
    unittest.main()
