import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.5


@dataclass
class DSACConfig:
    observation_dim: int
    action_dim: int = 2
    action_low: Tuple[float, float] = (-1.0, -1.0)
    action_high: Tuple[float, float] = (2.0, 1.0)
    hidden_dim: int = 256
    num_quantiles: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200000
    learning_starts: int = 10000
    train_freq: int = 10
    gradient_steps: int = 10
    seed: int = 0


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DSACActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim * 2, hidden_dim)
        self.mean = nn.Linear(hidden_dim * 2, action_dim)
        self.log_std = nn.Linear(hidden_dim * 2, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample_unit(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(std)
        unit_action = torch.tanh(z)
        normal_log_prob = -0.5 * (((z - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - unit_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return unit_action, log_prob

    def unit_std(self, obs: torch.Tensor) -> torch.Tensor:
        _, log_std = self(obs)
        return torch.exp(log_std)


class DistributionalCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_quantiles: int):
        super().__init__()
        self.num_quantiles = int(num_quantiles)
        self.q1 = _mlp(obs_dim + action_dim, num_quantiles, hidden_dim)
        self.q2 = _mlp(obs_dim + action_dim, num_quantiles, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def mean_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.min(q1.mean(dim=-1, keepdim=True), q2.mean(dim=-1, keepdim=True))


class DSACReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, seed: int = 0):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.size = int(size)
        self.ptr = 0
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.count, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class DSACPolicy:
    def __init__(self, config: DSACConfig, device: str = "auto"):
        self.config = config
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.actor = DSACActor(config.observation_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = DistributionalCritic(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.num_quantiles,
        ).to(self.device)
        self.critic_target = DistributionalCritic(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.num_quantiles,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.action_low = torch.as_tensor(config.action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(config.action_high, dtype=torch.float32, device=self.device)

    def _obs_tensor(self, obs: Any) -> torch.Tensor:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim > 1:
            obs_arr = obs_arr.reshape(obs_arr.shape[0], -1)
        else:
            obs_arr = obs_arr.reshape(1, -1)
        if obs_arr.shape[-1] != self.config.observation_dim:
            raise ValueError(f"DSAC expected obs_dim={self.config.observation_dim}, got {obs_arr.shape[-1]}")
        return torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)

    def _scale_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (unit_action + 1.0) * (self.action_high - self.action_low)

    def _unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.action_low) / (self.action_high - self.action_low + 1e-8) - 1.0

    def sample_tensor(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        unit_action, log_prob = self.actor.sample_unit(obs, deterministic=deterministic)
        return self._scale_action(unit_action), log_prob

    def predict(self, obs: Any, deterministic: bool = True):
        with torch.no_grad():
            obs_tensor = self._obs_tensor(obs)
            action, _ = self.sample_tensor(obs_tensor, deterministic=deterministic)
        action_np = action.detach().cpu().numpy().astype(np.float32)
        return action_np, None

    def action_std(self, obs: Any) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = self._obs_tensor(obs)
            std = self.actor.unit_std(obs_tensor)
            scaled_std = 0.5 * std * (self.action_high - self.action_low)
        return scaled_std.detach().cpu().numpy().reshape(-1, self.config.action_dim)[0]

    def terminal_cost(self, obs: Any, action: Any) -> float:
        with torch.no_grad():
            obs_tensor = self._obs_tensor(obs)
            action_tensor = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1), device=self.device)
            q_return = self.critic.mean_q(obs_tensor, action_tensor)
        return float(-q_return.detach().cpu().numpy().reshape(-1)[0])

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            os.path.join(path, "model.pt"),
        )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "DSACPolicy":
        if path.endswith(".pt"):
            path = os.path.dirname(path)
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = DSACConfig(**json.load(f))
        policy = cls(config, device=device)
        payload = torch.load(os.path.join(path, "model.pt"), map_location=policy.device)
        policy.actor.load_state_dict(payload["actor"])
        policy.critic.load_state_dict(payload["critic"])
        policy.critic_target.load_state_dict(payload.get("critic_target", payload["critic"]))
        return policy


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_quantiles = pred.shape[-1]
    taus = (torch.arange(num_quantiles, device=pred.device, dtype=torch.float32) + 0.5) / num_quantiles
    diff = target.unsqueeze(1) - pred.unsqueeze(2)
    huber = F.smooth_l1_loss(pred.unsqueeze(2), target.unsqueeze(1), reduction="none")
    weight = torch.abs(taus.view(1, num_quantiles, 1) - (diff.detach() < 0.0).float())
    return (weight * huber).mean()


class DSACTrainer:
    def __init__(self, policy: DSACPolicy, env: Any, config: DSACConfig, tensorboard_log: Optional[str] = None):
        self.policy = policy
        self.env = env
        self.config = config
        self.buffer = DSACReplayBuffer(config.observation_dim, config.action_dim, config.buffer_size, config.seed)
        self.actor_opt = torch.optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr)
        self.writer = None
        if tensorboard_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                os.makedirs(tensorboard_log, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_log)
                print(f"TensorBoard logs: {tensorboard_log}")
            except Exception as exc:
                print(f"TensorBoard disabled: {exc}")

    def learn(self, total_timesteps: int, save_dir: str, log_interval: int = 1000, episode_log_interval: int = 100) -> DSACPolicy:
        obs = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        recent_rewards = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_actor_losses = deque(maxlen=100)
        recent_critic_losses = deque(maxlen=100)
        recent_q_values = deque(maxlen=100)
        recent_log_probs = deque(maxlen=100)
        recent_successes = deque(maxlen=100)
        recent_collisions = deque(maxlen=100)
        recent_min_laser = deque(maxlen=100)
        recent_abs_frenet_d = deque(maxlen=100)
        best_score = -float("inf")
        start_time = time.time()
        for step in range(1, int(total_timesteps) + 1):
            if step < self.config.learning_starts:
                action = np.asarray([self.env.action_space.sample()]).reshape(-1)
            else:
                action, _ = self.policy.predict(obs, deterministic=False)
                action = action.reshape(-1)
            next_obs, rewards, dones, infos = self.env.step(action.reshape(1, -1))
            reward = float(np.asarray(rewards).reshape(-1)[0])
            done = bool(np.asarray(dones).reshape(-1)[0])
            info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            next_obs_flat = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            self.buffer.add(obs, action, reward, next_obs_flat, done)
            obs = next_obs_flat
            episode_reward += reward
            episode_length += 1
            if done:
                episode_count += 1
                recent_rewards.append(float(episode_reward))
                recent_lengths.append(int(episode_length))
                recent_successes.append(float(bool(info.get("is_success", False))))
                recent_collisions.append(float(bool(info.get("is_collision", False))))
                recent_min_laser.append(float(info.get("min_laser_dist", 0.0)))
                recent_abs_frenet_d.append(abs(float(info.get("frenet_d", 0.0))))
                curriculum_stats = self._record_curriculum_episode(info)
                self._write_curriculum_metrics(curriculum_stats, step)
                self._write_scalar("rollout/ep_rew", episode_reward, step)
                self._write_scalar("rollout/ep_len", episode_length, step)
                self._write_scalar("rollout/episodes", episode_count, step)
                score = self._model_selection_score(
                    recent_rewards,
                    recent_successes,
                    recent_collisions,
                    recent_abs_frenet_d,
                )
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    self.policy.save(os.path.join(save_dir, "best_model"))
                    self._write_scalar("rollout/best_score", best_score, step)
                if episode_log_interval > 0 and episode_count % int(episode_log_interval) == 0:
                    print(self._format_episode_summary_table(
                        step=step,
                        total_timesteps=int(total_timesteps),
                        episode_count=episode_count,
                        curriculum_stats=curriculum_stats,
                        recent_rewards=recent_rewards,
                        recent_lengths=recent_lengths,
                        recent_successes=recent_successes,
                        recent_collisions=recent_collisions,
                        recent_min_laser=recent_min_laser,
                        recent_abs_frenet_d=recent_abs_frenet_d,
                        recent_actor_losses=recent_actor_losses,
                        recent_critic_losses=recent_critic_losses,
                        recent_q_values=recent_q_values,
                        recent_log_probs=recent_log_probs,
                        best_score=best_score,
                    ), flush=True)
                obs = np.asarray(self.env.reset(), dtype=np.float32).reshape(-1)
                episode_reward = 0.0
                episode_length = 0
            if self.buffer.count >= self.config.learning_starts and step % self.config.train_freq == 0:
                for _ in range(self.config.gradient_steps):
                    metrics = self.update()
                    recent_actor_losses.append(metrics["actor_loss"])
                    recent_critic_losses.append(metrics["critic_loss"])
                    recent_q_values.append(metrics["mean_q"])
                    recent_log_probs.append(metrics["mean_log_prob"])
                    self._write_training_metrics(metrics, step)
            if log_interval > 0 and step % log_interval == 0:
                fps = step / max(time.time() - start_time, 1e-6)
                self._write_scalar("time/fps", fps, step)
                self._write_scalar("train/buffer_size", self.buffer.count, step)
                self._write_scalar("rollout/ep_rew_mean", self._mean_or_nan(recent_rewards), step)
                self._write_scalar("rollout/ep_len_mean", self._mean_or_nan(recent_lengths), step)
                self._write_scalar("rollout/current_ep_reward", episode_reward, step)
                self._write_scalar("rollout/current_ep_len", episode_length, step)
                self._write_scalar("rollout/success_rate", self._mean_or_nan(recent_successes), step)
                self._write_scalar("rollout/collision_rate", self._mean_or_nan(recent_collisions), step)
                self._write_scalar("rollout/mean_min_laser_dist", self._mean_or_nan(recent_min_laser), step)
                self._write_scalar("rollout/mean_abs_frenet_d", self._mean_or_nan(recent_abs_frenet_d), step)
                print(self._format_log_line(
                    step=step,
                    total_timesteps=int(total_timesteps),
                    elapsed=time.time() - start_time,
                    episode_count=episode_count,
                    current_episode_reward=episode_reward,
                    current_episode_length=episode_length,
                    recent_rewards=recent_rewards,
                    recent_lengths=recent_lengths,
                    recent_actor_losses=recent_actor_losses,
                    recent_critic_losses=recent_critic_losses,
                    recent_q_values=recent_q_values,
                    recent_log_probs=recent_log_probs,
                    recent_successes=recent_successes,
                    recent_collisions=recent_collisions,
                    recent_min_laser=recent_min_laser,
                    recent_abs_frenet_d=recent_abs_frenet_d,
                ))
            if step % max(log_interval, 1) == 0:
                self._flush_writer()
        self._flush_writer()
        return self.policy

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _write_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is None or not np.isfinite(float(value)):
            return
        self.writer.add_scalar(tag, float(value), int(step))

    def _write_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._write_scalar("train/actor_loss", metrics["actor_loss"], step)
        self._write_scalar("train/critic_loss", metrics["critic_loss"], step)
        self._write_scalar("train/mean_q", metrics["mean_q"], step)
        self._write_scalar("train/mean_log_prob", metrics["mean_log_prob"], step)
        self._write_scalar("train/alpha", self.config.alpha, step)

    def _write_curriculum_metrics(self, stats: Dict[str, float], step: int) -> None:
        for key, value in stats.items():
            self._write_scalar(f"curriculum/{key}", value, step)

    def _record_curriculum_episode(self, info: Dict[str, Any]) -> Dict[str, float]:
        env = self.env
        visited = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            manager = getattr(env, "curriculum_manager", None)
            if manager is not None:
                stats = manager.record_episode(info)
                self._set_curriculum_stage(manager.stage)
                return stats
            env = getattr(env, "env", None) or getattr(env, "venv", None)
            if hasattr(env, "envs") and env.envs:
                env = env.envs[0]
        return {"stage": 0.0, "success_rate": 0.0, "collision_rate": 0.0, "mean_abs_frenet_d": 0.0, "mean_min_laser_dist": 0.0}

    def _set_curriculum_stage(self, stage: int) -> None:
        env = self.env
        visited = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            if hasattr(env, "curriculum_stage"):
                env.curriculum_stage = int(stage)
                return
            env = getattr(env, "env", None) or getattr(env, "venv", None)
            if hasattr(env, "envs") and env.envs:
                env = env.envs[0]

    def _model_selection_score(self, rewards, successes, collisions, abs_frenet_d) -> float:
        reward_mean = self._mean_or_nan(rewards)
        success_rate = self._mean_or_nan(successes)
        collision_rate = self._mean_or_nan(collisions)
        mean_abs_d = self._mean_or_nan(abs_frenet_d)
        if not np.isfinite(reward_mean):
            return -float("inf")
        if not np.isfinite(success_rate):
            success_rate = 0.0
        if not np.isfinite(collision_rate):
            collision_rate = 1.0
        if not np.isfinite(mean_abs_d):
            mean_abs_d = 3.0
        return float(reward_mean + 1500.0 * success_rate - 800.0 * collision_rate - 60.0 * mean_abs_d)

    def _flush_writer(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def _mean_or_nan(self, values) -> float:
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _format_float(self, value: float, digits: int = 3) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"

    def _format_episode_summary_table(
        self,
        step: int,
        total_timesteps: int,
        episode_count: int,
        curriculum_stats: Dict[str, float],
        recent_rewards,
        recent_lengths,
        recent_successes,
        recent_collisions,
        recent_min_laser,
        recent_abs_frenet_d,
        recent_actor_losses,
        recent_critic_losses,
        recent_q_values,
        recent_log_probs,
        best_score: float,
    ) -> str:
        stage = int(curriculum_stats.get("stage", 0.0))
        rows = [
            ("progress/step", f"{step}/{total_timesteps}"),
            ("rollout/episodes", episode_count),
            ("curriculum/stage", stage),
            ("rollout/ep_rew_mean", self._format_float(self._mean_or_nan(recent_rewards), 2)),
            ("rollout/ep_len_mean", self._format_float(self._mean_or_nan(recent_lengths), 1)),
            ("rollout/success_rate", self._format_float(self._mean_or_nan(recent_successes), 3)),
            ("rollout/collision_rate", self._format_float(self._mean_or_nan(recent_collisions), 3)),
            ("rollout/mean_min_laser_dist", self._format_float(self._mean_or_nan(recent_min_laser), 2)),
            ("rollout/mean_abs_frenet_d", self._format_float(self._mean_or_nan(recent_abs_frenet_d), 2)),
            ("train/actor_loss", self._format_float(self._mean_or_nan(recent_actor_losses))),
            ("train/critic_loss", self._format_float(self._mean_or_nan(recent_critic_losses))),
            ("train/mean_q", self._format_float(self._mean_or_nan(recent_q_values))),
            ("train/mean_log_prob", self._format_float(self._mean_or_nan(recent_log_probs))),
            ("rollout/best_score", self._format_float(best_score, 2)),
        ]
        key_width = max(len(str(key)) for key, _ in rows)
        value_width = max(len(str(value)) for _, value in rows)
        border = "-" * (key_width + value_width + 7)
        title = f"Episode summary after {episode_count} episodes"
        lines = [title, border]
        lines.extend(f"| {key:<{key_width}} | {str(value):>{value_width}} |" for key, value in rows)
        lines.append(border)
        return "\n".join(lines)

    def _format_log_line(
        self,
        step: int,
        total_timesteps: int,
        elapsed: float,
        episode_count: int,
        current_episode_reward: float,
        current_episode_length: int,
        recent_rewards,
        recent_lengths,
        recent_actor_losses,
        recent_critic_losses,
        recent_q_values,
        recent_log_probs,
        recent_successes,
        recent_collisions,
        recent_min_laser,
        recent_abs_frenet_d,
    ) -> str:
        fps = step / max(elapsed, 1e-6)
        rows = [
            ("time/iterations", step),
            ("time/total_timesteps", f"{step}/{total_timesteps}"),
            ("time/fps", f"{fps:.1f}"),
            ("rollout/episodes", episode_count),
            ("rollout/ep_rew_mean", self._format_float(self._mean_or_nan(recent_rewards), 2)),
            ("rollout/ep_len_mean", self._format_float(self._mean_or_nan(recent_lengths), 1)),
            ("rollout/current_ep_reward", f"{current_episode_reward:.2f}"),
            ("rollout/current_ep_len", current_episode_length),
            ("rollout/success_rate", self._format_float(self._mean_or_nan(recent_successes), 3)),
            ("rollout/collision_rate", self._format_float(self._mean_or_nan(recent_collisions), 3)),
            ("rollout/mean_min_laser_dist", self._format_float(self._mean_or_nan(recent_min_laser), 2)),
            ("rollout/mean_abs_frenet_d", self._format_float(self._mean_or_nan(recent_abs_frenet_d), 2)),
            ("train/buffer_size", self.buffer.count),
            ("train/actor_loss", self._format_float(self._mean_or_nan(recent_actor_losses))),
            ("train/critic_loss", self._format_float(self._mean_or_nan(recent_critic_losses))),
            ("train/mean_q", self._format_float(self._mean_or_nan(recent_q_values))),
            ("train/mean_log_prob", self._format_float(self._mean_or_nan(recent_log_probs))),
            ("train/alpha", f"{self.config.alpha:.3f}"),
        ]
        key_width = max(len(str(key)) for key, _ in rows)
        value_width = max(len(str(value)) for _, value in rows)
        border = "-" * (key_width + value_width + 7)
        lines = [border]
        lines.extend(f"| {key:<{key_width}} | {str(value):>{value_width}} |" for key, value in rows)
        lines.append(border)
        return "\n".join(lines)

    def update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.config.batch_size, self.policy.device)
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample_tensor(batch["next_obs"], deterministic=False)
            next_q1, next_q2 = self.policy.critic_target(batch["next_obs"], next_action)
            next_q = torch.min(next_q1, next_q2) - self.config.alpha * next_log_prob
            target = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * next_q
        q1, q2 = self.policy.critic(batch["obs"], batch["actions"])
        critic_loss = quantile_huber_loss(q1, target) + quantile_huber_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_prob = self.policy.sample_tensor(batch["obs"], deterministic=False)
        actor_q = self.policy.critic.mean_q(batch["obs"], new_action)
        actor_loss = (self.config.alpha * log_prob - actor_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
                target_param.data.mul_(1.0 - self.config.tau).add_(self.config.tau * param.data)
            mean_q = actor_q.mean()
            mean_log_prob = log_prob.mean()
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "mean_q": float(mean_q.detach().cpu().item()),
            "mean_log_prob": float(mean_log_prob.detach().cpu().item()),
        }


def parse_dsac_args():
    parser = argparse.ArgumentParser(description="Train a DSAC policy for USV navigation.")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--model-name", default="dsac_high_level", help="Name used under data/<model-name>/ for outputs.")
    parser.add_argument("--save-dir", default=None, help="Model output directory. Defaults to data/<model-name>/models.")
    parser.add_argument("--log-dir", default=None, help="Monitor log directory. Defaults to data/<model-name>/logs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--control-mode", choices=("low_level_velocity", "high_level_frenet"), default="high_level_frenet")
    parser.add_argument("--dynamics-model", choices=("ideal", "inertia"), default="ideal")
    parser.add_argument("--curriculum", choices=("off", "auto"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--episode-log-interval", type=int, default=100)
    parser.add_argument("--tensorboard-log", default=None, help="TensorBoard event directory. Defaults to <log-dir>/tensorboard.")
    return parser.parse_args()
