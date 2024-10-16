import unittest
from unittest.mock import MagicMock
from replaybuffer.background_saver import BackgroundSaver
from replaybuffer.disk_manager import DiskManager
import time

class TestBackgroundSaver(unittest.TestCase):
    def setUp(self):
        self.disk_manager = MagicMock(spec=DiskManager)
        self.background_saver = BackgroundSaver(self.disk_manager, batch_size=10)

    def test_run_and_stop(self):
        self.background_saver.run()
        self.assertTrue(self.background_saver.running)
        self.background_saver.stop()
        self.assertFalse(self.background_saver.running)

    def test_save(self):
        self.background_saver.run()
        experience = {"state": [1, 2, 3], "action": 1, "reward": 1.0, "next_state": [4, 5, 6], "done": False}
        self.background_saver.save(experience)
        time.sleep(0.1)  # Give some time for the background thread to process
        self.disk_manager.save_to_disk.assert_called_with(experience)
        self.background_saver.stop()

    def test_process(self):
        self.background_saver.run()
        experiences = [
            {"state": [1, 2, 3], "action": 1, "reward": 1.0, "next_state": [4, 5, 6], "done": False},
            {"state": [7, 8, 9], "action": 2, "reward": 2.0, "next_state": [10, 11, 12], "done": True}
        ]
        for exp in experiences:
            self.background_saver.save(exp)
        
        time.sleep(0.1)  # Give some time for the background thread to process
        calls = [unittest.mock.call(exp) for exp in experiences]
        self.disk_manager.save_to_disk.assert_has_calls(calls, any_order=True)
        self.background_saver.stop()

if __name__ == '__main__':
    unittest.main()