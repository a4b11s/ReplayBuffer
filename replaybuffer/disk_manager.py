import os
import logging
import numpy as np
import h5py as h5

from replaybuffer.experience import Experience


class DiskManager:
    disk_pointer = 0
    length = 0

    def __init__(self, h5_path, max_size, lock):
        self.logger = logging.getLogger("DiskManager")

        self.h5_path = h5_path
        self.max_size = max_size
        self.lock = lock

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
                    dtype=np.float32,
                )

        self.logger.debug("HDF5 file initialized")

    def save_to_disk(self, data: dict):
        with self.lock:
            with h5.File(self.h5_path, "a") as h5_file:
                for key, value in data.items():
                    h5_file[key][
                        self.disk_pointer : self.disk_pointer + len(value)
                    ] = value

        self.disk_pointer = (self.disk_pointer + len(value)) % self.max_size

        self.length += len(value)
        if self.length > self.max_size:
            self.length = self.max_size

    def load_batch_from_disk(self, indices):
        with self.lock:
            with h5.File(self.h5_path, "r") as h5_file:
                return {key: h5_file[key][indices] for key in h5_file.keys()}


if __name__ == "__main__":
    DiskManager("", 100, None)
