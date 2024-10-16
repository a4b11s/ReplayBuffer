import logging
import queue
import threading

from replaybuffer.disk_manager import DiskManager


class BackgroundSaver:
    def __init__(self, disk_manager, batch_size, queue_size=None):
        if queue_size is None:
            queue_size = batch_size * 2

        self.logger = logging.getLogger("BackgroundSaver")
        self.disk_manager: DiskManager = disk_manager
        self.batch_size = batch_size
        self.save_queue = queue.Queue(maxsize=queue_size)

        self.thread = threading.Thread(target=self._process)
        self.thread.daemon = True
        self.running = False

    def run(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.save_queue.put(None)  # Sentinel value to indicate stopping
        self.thread.join()  # Wait for the thread to finish

    def save(self, experience):
        self.logger.debug("Saving experience to disk")
        self.save_queue.put(experience)

    def _process(self):
        buffer = []
        while True:
            try:
                experience = self.save_queue.get(timeout=3)

                if experience is None:  # Check for sentinel to stop processing
                    if buffer:
                        self.logger.debug("Flushing remaining experiences to disk")
                        self.disk_manager.save_to_disk(buffer)
                    break

                buffer.append(experience)

                if len(buffer) >= self.save_queue.maxsize / 2:
                    self.logger.debug(
                        f"Saving batch of {len(buffer)} experiences to disk"
                    )
                    if len(buffer):
                        self.disk_manager.save_to_disk(buffer)
                        buffer.clear()

            except queue.Empty:
                self.logger.debug(
                    f"Timeout, Saving batch of {len(buffer)} experiences to disk"
                )
                self.disk_manager.save_to_disk(buffer)
                buffer.clear()
                continue  # Timeout reached, continue processing

            except Exception as e:
                self.logger.error(f"Error while saving experience: {e}")
                break

        # If there are any remaining experiences, flush them to disk before stopping.
        if buffer:
            self.logger.debug(
                "Flushing remaining experiences to disk after exiting loop"
            )
            self.disk_manager.save_to_disk(buffer)
