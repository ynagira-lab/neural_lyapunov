"""Train the Residual SAC controller used in the Neural Lyapunov experiments.

Based on day05_compare.py. The nonlinear plant, environment, cost, SAC
hyperparameters, and random seed are preserved. This script trains only
Residual SAC, without the standalone SAC or diagnostic plots.

Run: python residual_sac_training.py
Output: out/residual_sac_nonlinear_100k.zip
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_discrete_are
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

TOTAL_TIMESTEPS = 100_000
SEED = 0
MAX_STEPS = 100
DU_MAX = 0.5
P_FAIL = 5.0
V_FAIL = 10.0
FAILURE_PENALTY = 1000.0

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
Q = np.diag([1.0, 0.1])
R = np.array([[0.01]])
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def nonlinear_term(x):
    p = float(x[0])
    return np.array([0.0, 0.02 * p**3])


def plant_step(x, u):
    return A @ x + B[:, 0] * u + nonlinear_term(x)


def stage_cost(x, u):
    return float(x @ Q @ x + R[0, 0] * u**2)


def is_failure(x):
    if not np.all(np.isfinite(x)):
        return True
    if abs(x[0]) > P_FAIL:
        return True
    if abs(x[1]) > V_FAIL:
        return True
    return False


def make_finite_state(x):
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(x, -10.0, 10.0)


class ResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.max_steps = MAX_STEPS
        self.action_space = spaces.Box(
            low=np.array([-DU_MAX], dtype=np.float32),
            high=np.array([DU_MAX], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-P_FAIL, -V_FAIL], dtype=np.float32),
            high=np.array([P_FAIL, V_FAIL], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [self.np_random.uniform(-1.5, 1.5),
             self.np_random.uniform(-0.5, 0.5)],
            dtype=np.float64,
        )
        self.step_count = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        x = self.state.copy()
        du = float(np.clip(action[0], -DU_MAX, DU_MAX))
        u_lqr = float(-(K @ x)[0])
        u = u_lqr + du
        cost = stage_cost(x, u)
        x_next = plant_step(x, u)
        self.step_count += 1
        failure = is_failure(x_next)
        if failure:
            x_next = make_finite_state(x_next)
            reward = -FAILURE_PENALTY
            terminated = True
        else:
            reward = -cost
            terminated = False
        truncated = self.step_count >= self.max_steps
        self.state = x_next
        info = {"failure": failure, "du": du, "u_lqr": u_lqr,
                "u": u, "stage_cost": cost}
        return x_next.astype(np.float32), float(reward), terminated, truncated, info


def main():
    print("Training Residual SAC (100,000 steps, seed=0)")
    vec_env = DummyVecEnv([lambda: Monitor(ResidualEnv())])
    model = SAC(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=SEED,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    output = Path(__file__).resolve().parent / "out" / "residual_sac_nonlinear_100k"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"Saved: {output}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
