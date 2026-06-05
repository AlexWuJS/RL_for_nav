from collections import deque
from typing import Any

import numpy as np


class PolicyObservationStacker:
    """Build the flat frame stack expected by a trained DSAC policy."""

    def __init__(self, expected_dim: int):
        self.expected_dim = int(expected_dim)
        self.frame_dim = None
        self.stack_size = 1
        self.frames = deque()

    def reset(self, obs: Any) -> np.ndarray:
        frame = self._flatten(obs)
        self._configure(frame)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame.copy())
        return self.current()

    def update(self, obs: Any) -> np.ndarray:
        frame = self._flatten(obs)
        self._configure(frame)
        self.frames.append(frame.copy())
        while len(self.frames) > self.stack_size:
            self.frames.popleft()
        return self.current()

    def current(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Observation stack is empty; call reset() first.")
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)

    def _configure(self, frame: np.ndarray) -> None:
        frame_dim = int(frame.shape[0])
        if frame_dim == self.expected_dim:
            self.frame_dim = frame_dim
            self.stack_size = 1
            return
        if self.expected_dim % frame_dim != 0:
            raise ValueError(f"Policy expected obs_dim={self.expected_dim}, but environment returned obs_dim={frame_dim}.")
        self.frame_dim = frame_dim
        self.stack_size = self.expected_dim // frame_dim

    @staticmethod
    def _flatten(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)
