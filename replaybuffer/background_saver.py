import queue
import threading

from replaybuffer.disk_manager import DiskManager


class BackgroundSaver:
    def __init__(self, disk_manager, batch_size):
        self.disk_manager: DiskManager = disk_manager

        self.batch_size = batch_size

        self.save_queue = queue.Queue(maxsize=50)
        self.thread = threading.Thread(target=self._process)
        self.thread.daemon = True
        self.running = False

    def run(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def save(self, experience):
        self.save_queue.put(experience)

    def _process(self):
        while self.running:
            if not self.save_queue.empty():
                experience = self.save_queue.get()
                self.disk_manager.save_to_disk(experience)
