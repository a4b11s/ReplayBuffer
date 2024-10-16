import os
import tempfile
import unittest
import h5py as h5
import numpy as np
from threading import Lock
from replaybuffer.disk_manager import DiskManager


class TestDiskManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "test.h5")
        self.max_size = 100
        self.lock = Lock()
        self.disk_manager = DiskManager(self.h5_path, self.max_size, self.lock)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_h5_file(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        with h5.File(self.h5_path, "r") as h5_file:
            self.assertIn("data", h5_file)
            self.assertEqual(h5_file["data"].shape, (self.max_size, 10, 10))

    def test_save_to_disk(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        with h5.File(self.h5_path, "r") as h5_file:
            np.testing.assert_array_equal(h5_file["data"][:5], data["data"])

    def test_save_to_disk_n_times(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}

        for _ in range(10):
            self.disk_manager.save_to_disk(data)

        with h5.File(self.h5_path, "r") as h5_file:
            np.testing.assert_array_equal(
                h5_file["data"][:50], np.tile(data["data"], (10, 1, 1))
            )

    def test_save_to_disk_wrap(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(100, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        with h5.File(self.h5_path, "r") as h5_file:
            np.testing.assert_array_equal(h5_file["data"][:100], data["data"])
            
    
    def test_length(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.length, 5)

    def test_length_n_times(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}

        for _ in range(10):
            self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.length, 50)
        
    def test_length_wrap(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(100, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.length, 100)

    def test_disk_pointer(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.disk_pointer, 5)

        for _ in range(10):
            self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.disk_pointer, 55)
    
    def test_disk_pointer_wrap(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(100, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        self.assertEqual(self.disk_manager.disk_pointer, 0)

    def test_load_batch_from_disk(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        indices = [0, 2, 4]
        loaded_data = self.disk_manager.load_batch_from_disk(indices)

        np.testing.assert_array_equal(loaded_data["data"], data["data"][indices])

    def test_load_batch_from_disk_n_times(self):
        shapes = {"data": (10, 10)}
        self.disk_manager._init_h5_file(shapes)

        data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}
        self.disk_manager.save_to_disk(data)

        indices = [0, 2, 4]
        for _ in range(10):
            loaded_data = self.disk_manager.load_batch_from_disk(indices)

            np.testing.assert_array_equal(loaded_data["data"], data["data"][indices])


if __name__ == "__main__":
    unittest.main()
