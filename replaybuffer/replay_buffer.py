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
        logging.debug("Initializing ReplayBuffer")
        self.max_size = max_size
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.prefetch_batches = queue.Queue(maxsize=prefetch)
        self.save_queue = queue.Queue(maxsize=prefetch)
        self.device = device

        self.length = 0

        logging.info(
            "ReplayBuffer initialized with max_size=%d, h5_path=%s, image_shape=%s, device=%s, batch_size=%d, prefetch=%d",
            max_size,
            h5_path,
            image_shape,
            device,
            batch_size,
            prefetch,
        )

        self.disk_pointer = 0

        self.lock = filelock.FileLock(h5_path + ".lock")
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.save_thread = threading.Thread(target=self._prefetch_save)

        self._init_h5_file()
        self._run_threads()

    def add(self, state, action, reward, next_state, done):
        logging.debug("Adding experience to ReplayBuffer")
        state = self._add_prepare(state)
        action = self._add_prepare(action)
        reward = self._add_prepare(reward)
        next_state = self._add_prepare(next_state)
        done = self._add_prepare(done)

        experience = np.array([[state, action, reward, next_state, done]], dtype=object)
        self._save_batch_to_disk(experience)

    def sample(self):
        logging.debug("Sampling from ReplayBuffer")
        print(self.prefetch_batches.qsize())
        return self.prefetch_batches.get()

    def __len__(self):
        logging.debug("Getting length of ReplayBuffer")
        return self.length

    def _add_prepare(self, data):
        logging.debug("Preparing data for addition to ReplayBuffer")
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data

    def __getitem__(self, idx):
        logging.debug("Getting item from ReplayBuffer at index %d", idx)
        return self.h5_file["experiences"][idx]

    def _prefetch(self):
        logging.debug("Starting prefetch thread")
        while True:
            if not self.prefetch_batches.full() and self.length >= self.batch_size:
                loaded_batch = self._load_batch_from_disk()
                sampled = []

                for exp in loaded_batch:
                    sampled.append(
                        [torch.tensor(field, device=self.device) for field in exp]
                    )

                self.prefetch_batches.put(sampled)

    def _prefetch_save(self):
        logging.debug("Starting prefetch save thread")
        while True:
            if not self.save_queue.empty():
                slice, data = self.save_queue.get()
                self._save_to_disk(slice, data)

    def _save_batch_to_disk(self, batch):
        logging.debug("Saving batch to disk")
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

    def _save_to_disk(self, slice, data):
        logging.debug("Saving data to disk at slice %s", slice)

        with self.lock:
            logging.debug("Acquired lock in save_to_disk")
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

        logging.debug("Saved data to disk at slice %s, and lock realized", slice)

    def _load_batch_from_disk(self):
        logging.debug("Loading batch from disk")
        if self.length < self.batch_size:
            raise ValueError("Not enough samples in the buffer")

        indx = np.sort(np.random.choice(self.length, self.batch_size, replace=False))

        with self.lock:
            logging.debug("Acquired lock in load_batch_from_disk")
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

        logging.debug("Loaded batch from disk and lock released")
        return list(zip(states, actions, rewards, next_states, dones))

    def _init_h5_file(self):
        logging.debug("Initializing HDF5 file")
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
                dtype=np.bool,
            )

    def _run_threads(self):
        logging.debug("Running prefetch and save threads")
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        self.save_thread.daemon = True
        self.save_thread.start()
