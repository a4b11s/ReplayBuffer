import logging
import filelock

from replaybuffer.disk_manager import DiskManager
from replaybuffer.prefetcher import Prefetcher
from replaybuffer.background_saver import BackgroundSaver


class ReplayBuffer:
    def __init__(self, max_size, h5_path, image_shape, device, batch_size):
        self.logger = logging.getLogger("ReplayBuffer")
        self.max_size = max_size
        self.h5_path = h5_path
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.lock = filelock.FileLock(h5_path + ".lock")

        self.disk_manager = DiskManager(h5_path, max_size, self.lock)
        self.prefetcher = Prefetcher(self.disk_manager, device, batch_size)
        self.background_saver = BackgroundSaver(self.disk_manager, batch_size)

        self._init_h5_file()
        self.start_subprocesses()

    def _init_h5_file(self):
        shapes = {
            "state": self.image_shape,
            "action": (1,),
            "reward": (1,),
            "next_state": self.image_shape,
            "done": (1,),
        }
        self.disk_manager._init_h5_file(shapes)

    def start_subprocesses(self):
        self.prefetcher.start()
        self.background_saver.start()

    def add(self, state, action, reward, next_state, done):
        self.background_saver.save(
            {
                "state": state,
                "action": [action],
                "reward": [reward],
                "next_state": next_state,
                "done": done,
            }
        )

    def sample(self):
        return self.prefetcher.sample()
    
    def __del__(self):
        self.background_saver.stop()
        self.prefetcher.stop()
        self.disk_manager.lock.release()
        self.lock.release()