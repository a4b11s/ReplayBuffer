import logging
import threading
import queue
import numpy as np

from replaybuffer.disk_manager import DiskManager
from replaybuffer.experience import Experience


class Prefetcher:
    def __init__(self, disk_manager, device, batch_size, prefetch_queue_size=50):
        self.logger = logging.getLogger("Prefetcher")
        self.disk_manager: DiskManager = disk_manager

        self.device = device
        self.batch_size = batch_size

        # Prefetch queue for batched data
        self.prefetch_batches = queue.Queue(maxsize=prefetch_queue_size)
        self.sampled_indices = queue.Queue(
            maxsize=prefetch_queue_size * 2
        )  # Pre-sample indices
        self.running = False

        # Prefetching and sampling threads
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.sampling_thread = threading.Thread(target=self._sample_batches)
        self.prefetch_thread.daemon = True
        self.sampling_thread.daemon = True

    def run(self):
        """Start the prefetch and sampling threads."""
        self.running = True
        self.sampling_thread.start()
        self.prefetch_thread.start()

    def stop(self):
        """Stop the threads gracefully."""
        self.running = False
        self.sampling_thread.join()
        self.prefetch_thread.join()

    def _sample_batches(self):
        """Sampling thread that prepares batch indices ahead of time."""
        while self.running:
            if not self.sampled_indices.full() and self.disk_manager.length > self.batch_size:
                self.logger.debug("Sampling batch indices")
                indices = np.sort(
                    np.random.choice(
                        self.disk_manager.length, self.batch_size, replace=False
                    )
                )
                self.sampled_indices.put(indices)
                self.logger.debug(f"Sampled batch indices: {indices}")

    def _prefetch(self):
        """Prefetch thread that loads batches of data from disk."""
        while self.running:
            if not self.prefetch_batches.full() and not self.sampled_indices.empty():
                self.logger.debug("Prefetching batch")
                indices = self.sampled_indices.get()  # Get pre-sampled indices
                loaded_batch = self.disk_manager.load_batch_from_disk(indices)

                # Optional: Move batch to device asynchronously (e.g., GPU)
                # loaded_batch = {k: v.to(self.device) for k, v in loaded_batch.items()}

                self.prefetch_batches.put(loaded_batch)
                self.logger.debug("Batch was pre-fetched and added to queue")

    def get_sample(self):
        """Retrieve a pre-fetched sample batch."""
        return self.prefetch_batches.get()
