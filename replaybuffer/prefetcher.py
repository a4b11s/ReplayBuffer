import threading
import queue
import numpy as np
import torch

from replaybuffer.disk_manager import DiskManager
from replaybuffer.experience import Experience


class Prefetcher:
    def __init__(self, disk_manager, device, batch_size):
        self.disk_manager: DiskManager = disk_manager

        self.device = device
        self.batch_size = batch_size

        self.prefetch_batches = queue.Queue(maxsize=50)
        self.thread = threading.Thread(target=self._process)
        self.thread.daemon = True
        self.running = False

    def run(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _process(self):
        while self.running:
            if (
                not self.prefetch_batches.full()
                and self.disk_manager.length > self.batch_size
            ):
                indicates = self._sample_batch_indicates()
                loaded_batch = self.disk_manager.load_batch_from_disk(indicates)
                sampled = [
                    [torch.tensor(field, device=self.device) for field in exp]
                    for exp in loaded_batch
                ]
                self.prefetch_batches.put(sampled)

    def get_sample(self):
        return self.prefetch_batches.get()

    def _sample_batch_indicates(self):
        return np.sort(np.random.randint(0, self.disk_manager.length, self.batch_size))
