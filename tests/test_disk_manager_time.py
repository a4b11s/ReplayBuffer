import os
import tempfile
import numpy as np
import threading
from replaybuffer.disk_manager import DiskManager

def test_time(repeat=1000):
    import time

    Lock = threading.Lock()

    temp_dir = tempfile.TemporaryDirectory()
    h5_path = os.path.join(temp_dir.name, "test.h5")

    disk_manager = DiskManager(h5_path, 100, Lock)

    shapes = {"data": (10, 10)}
    disk_manager._init_h5_file(shapes)
    data = {"data": np.random.rand(5, 10, 10).astype(np.float32)}

    start = time.time()
    for _ in range(repeat):
        disk_manager.save_to_disk(data)

    print(f"Time taken to save {repeat} times: ", time.time() - start)

    start = time.time()
    for _ in range(repeat):
        disk_manager.load_batch_from_disk([0, 2, 4])

    print(f"Time taken to load {repeat} times: ", time.time() - start)


if __name__ == "__main__":
    test_time(100_000)