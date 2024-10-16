from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
import numpy as np
import h5py as h5


class DiskManager:
    disk_pointer = 0
    length = 0

    def __init__(self, h5_path, max_size, lock, num_workers=4):
        self.logger = logging.getLogger("DiskManager")

        self.h5_path = h5_path
        self.max_size = max_size
        self.lock = lock

        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _init_h5_file(self, shapes: dict):
        self.logger.debug("Initializing HDF5 file")

        if not os.path.exists(self.h5_path):
            self.logger.debug("Creating directory for HDF5 file")
            os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)

        if os.path.exists(self.h5_path) and not h5.is_hdf5(self.h5_path):
            self.logger.critical("File exists but is not a valid HDF5 file")
            raise ValueError("File exists but is not a valid HDF5 file")

        with h5.File(self.h5_path, "w") as h5_file:
            for key, shape in shapes.items():
                h5_file.create_dataset(
                    key,
                    shape=(self.max_size, *shape),
                    maxshape=(self.max_size, *shape),
                    chunks=True,
                    compression="gzip",
                    dtype=np.float32,
                )

        self.logger.debug("HDF5 file initialized")

    def save_to_disk(self, data: dict):
        if len(data) == 0:
            return

        if isinstance(data, list):
            data = {key: np.array([exp[key] for exp in data]) for key in data[0].keys()}

        with self.lock:
            try:
                with h5.File(self.h5_path, "a") as h5_file:
                    for key, value in data.items():
                        h5_file[key][
                            self.disk_pointer : self.disk_pointer + len(value)
                        ] = value

            except Exception as e:
                self.logger.error(f"Error saving data to disk: {e}")
                self.logger.error(f"Data: {data}")

        self.disk_pointer = (self.disk_pointer + len(value)) % self.max_size

        self.length += min(self.length + len(value), self.max_size)

    def load_batch_from_disk(self, indices):
        with self.lock:
            with h5.File(self.h5_path, "r", swmr=True) as h5_file:
                self.logger.debug(f"Loading batch from indices")
                future_to_key = {
                    self.executor.submit(self._load_data, h5_file, key, indices): key
                    for key in h5_file.keys()
                }

                # Збирання результатів виконання паралельних завдань
                result = {}
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        result[key] = future.result()
                    except Exception as exc:
                        self.logger.error(f"Error loading key {key}: {exc}")

                return result

    def _load_data(self, h5_file, key, indices):
        return h5_file[key][indices]


if __name__ == "__main__":
    DiskManager("", 100, None)
