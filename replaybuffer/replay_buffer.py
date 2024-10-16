import os
import queue
import numpy as np
import torch
import filelock
import h5py as h5
import threading
import logging


class ReplayBuffer:
    def __init__(self, max_size, h5_path, image_shape, device, batch_size, prefetch=50):
        self.logger = logging.getLogger("ReplayBuffer")
        self.logger.debug("Initializing ReplayBuffer")

        # Child loggers for various functionalities
        self.thread_logger = self.logger.getChild("Thread")
        self.add_logger = self.logger.getChild("Add")
        self.sample_logger = self.logger.getChild("Sample")
        self.save_logger = self.logger.getChild("Save")
        self.load_logger = self.logger.getChild("Load")

        self.max_size = max_size
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.prefetch_batches = queue.Queue(maxsize=prefetch)
        self.save_queue = queue.Queue()
        self.device = device

        self.length = 0
        self.disk_pointer = 0

        self.lock = filelock.FileLock(h5_path + ".lock")
        self.thread = threading.Thread(target=self._process)

        self._init_h5_file()
        self.thread.start()  # Start the processing thread

    def add(self, state, action, reward, next_state, done):
        self.add_logger.debug("Adding experience to ReplayBuffer")

        state = self._add_prepare(state)
        action = self._add_prepare(action)
        reward = self._add_prepare(reward)
        next_state = self._add_prepare(next_state)
        done = self._add_prepare(done)

        self.add_logger.debug("Experience prepared for addition to ReplayBuffer")

        experience = np.array([[state, action, reward, next_state, done]], dtype=object)
        self._save_batch_to_disk(experience)

    def sample(self):
        self.sample_logger.debug("Sampling from ReplayBuffer")
        return self.prefetch_batches.get()

    def __len__(self):
        self.logger.debug("Getting length of ReplayBuffer")
        return self.length

    def _add_prepare(self, data):
        self.logger.debug("Preparing data for addition to ReplayBuffer")
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data

    def _process(self):
        self.thread_logger.debug("Starting processing thread")
        while True:
            if not self.prefetch_batches.full() and self.length >= self.batch_size:
                # Load a batch from disk if prefetch queue isn't full
                loaded_batch = self._load_batch_from_disk()
                sampled = []
                self.thread_logger.debug("Loaded batch from disk")
                for exp in loaded_batch:
                    sampled.append(
                        [torch.tensor(field, device=self.device) for field in exp]
                    )
                self.prefetch_batches.put(sampled)

                self.thread_logger.debug(
                    "Batch added to prefetch queue, length: %d",
                    self.prefetch_batches.qsize(),
                )
            if not self.save_queue.empty():
                # If there's data to save, process it
                self.save_logger.debug(
                    "Saving data to disk, queue length: %d", self.save_queue.qsize()
                )
                slice, data = self.save_queue.get()
                self._save_to_disk(slice, data)

    def _save_batch_to_disk(self, batch):
        self.save_logger.debug("Saving batch to disk")
        batch_size = len(batch)
        if self.disk_pointer + batch_size <= self.max_size:
            slice = (self.disk_pointer, self.disk_pointer + batch_size)
            self.disk_pointer += batch_size
        else:
            self.disk_pointer = 0
            slice = (self.disk_pointer, self.disk_pointer + batch_size)
            self.disk_pointer += batch_size

        self.save_queue.put((slice, batch))
        self.length += batch_size
        self.save_logger.debug("Batch saved to disk, new length: %d", self.length)

    def _save_to_disk(self, slice, data):
        self.save_logger.debug("Saving data to disk at slice %s", slice)

        with self.lock:
            self.save_logger.debug("Acquired lock in save_to_disk")
            with h5.File(self.h5_path, "a", libver="latest") as h5_file:
                h5_file["states"][slice[0] : slice[1]] = np.array(
                    [experience[0] for experience in data]
                )
                h5_file["actions"][slice[0] : slice[1]] = np.array(
                    [experience[1] for experience in data]
                )
                h5_file["rewards"][slice[0] : slice[1]] = np.array(
                    [experience[2] for experience in data]
                )
                h5_file["next_states"][slice[0] : slice[1]] = np.array(
                    [experience[3] for experience in data]
                )
                h5_file["dones"][slice[0] : slice[1]] = np.array(
                    [experience[4] for experience in data]
                )

        self.save_logger.debug(
            "Saved data to disk at slice %s, and lock released", slice
        )

    def _load_batch_from_disk(self):
        self.load_logger.debug("Loading batch from disk")
        if self.length < self.batch_size:
            raise ValueError("Not enough samples in the buffer")

        indx = np.sort(np.random.choice(self.length, self.batch_size, replace=False))

        with self.lock:
            self.load_logger.debug("Acquired lock in load_batch_from_disk")
            with h5.File(self.h5_path, "r") as h5_file:
                states = np.empty(
                    (self.batch_size, *h5_file["states"].shape[1:]),
                    dtype=h5_file["states"].dtype,
                )
                actions = np.empty((self.batch_size,), dtype=h5_file["actions"].dtype)
                rewards = np.empty((self.batch_size,), dtype=h5_file["rewards"].dtype)
                next_states = np.empty(
                    (self.batch_size, *h5_file["next_states"].shape[1:]),
                    dtype=h5_file["next_states"].dtype,
                )
                dones = np.empty((self.batch_size,), dtype=h5_file["dones"].dtype)

                # Fill the pre-allocated arrays
                for i, index in enumerate(indx):
                    states[i] = h5_file["states"][index]
                    actions[i] = h5_file["actions"][index]
                    rewards[i] = h5_file["rewards"][index]
                    next_states[i] = h5_file["next_states"][index]
                    dones[i] = h5_file["dones"][index]

        self.load_logger.debug("Loaded batch from disk and lock released")
        return list(zip(states, actions, rewards, next_states, dones))

    def _init_h5_file(self):
        self.logger.debug("Initializing HDF5 file")
        # TODO: Temporary fix. Force delete the file if it exists
        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)

        # Ensure the directory exists
        if not os.path.exists(self.h5_path):
            os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)
        # If the file exists but is not a valid HDF5 file, remove it
        if os.path.exists(self.h5_path) and not h5.is_hdf5(self.h5_path):
            os.remove(self.h5_path)

        state_shape = (self.max_size,) + self.image_shape
        action_shape = (self.max_size,)
        reward_shape = (self.max_size,)
        done_shape = (self.max_size,)

        with h5.File(self.h5_path, "w") as h5_file:
            h5_file.create_dataset(
                "states",
                shape=state_shape,
                maxshape=state_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "next_states",
                shape=state_shape,
                maxshape=state_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "actions",
                shape=action_shape,
                maxshape=action_shape,
                chunks=True,
                dtype=np.int64,  # Assuming integer actions
            )
            h5_file.create_dataset(
                "rewards",
                shape=action_shape,
                maxshape=reward_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "dones",
                shape=action_shape,
                maxshape=done_shape,
                chunks=True,
                dtype=np.bool_,
            )
        self.logger.debug("HDF5 file initialized")
