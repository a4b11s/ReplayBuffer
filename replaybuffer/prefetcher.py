import logging
import threading
import queue
import numpy as np

from replaybuffer.disk_manager import DiskManager
from replaybuffer.experience import Experience


class Prefetcher:
    def __init__(self, disk_manager, device, batch_size):
        self.logger = logging.getLogger("Prefetcher")
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
                and self.disk_manager.length >= self.batch_size
            ):
                self.logger.debug("Prefetching batch")
                indicates = self._sample_batch_indicates()
                loaded_batch = self.disk_manager.load_batch_from_disk(indicates)
                self.prefetch_batches.put(loaded_batch)
                self.logger.debug("Batch was put in queue")

    def get_sample(self):
        return self.prefetch_batches.get()

    def _sample_batch_indicates(self):
        self.logger.debug("Sampling batch indices")
        # Sample random non-repeating indices
        return np.sort(np.random.choice(self.disk_manager.length, self.batch_size, replace=False))