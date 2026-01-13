"""training/train_policy.py

Train a PPO policy (stable-baselines3) to guide RLNMC backbone selection.

This script trains on the *same setting as the paper-style experiments*:
- The environment simulates RLNMC steps: SA warm-up to a local minimum, then repeated
  NMC jumps where the action is a MultiBinary(N) mask selecting the backbone.
- The reward is the *improvement of the best energy seen so far* in the episode.

Design goals
------------
- Reproducible: deterministic seeds where possible.
- Framework-like: minimal assumptions about the rest of the project.
- Compatible with stable-baselines3 PPO + MlpPolicy.

Speed-oriented notes (NO quality change)
----------------------------------------
- On macOS, SubprocVecEnv often adds large overhead due to spawn/forkserver.
  By default we use DummyVecEnv ("auto") unless you explicitly request subproc.
- Tensorboard logging can be disabled to avoid overhead in smoke runs.
- You can control torch thread count for better CPU behavior.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Project imports
from env.sat_env import RLNMCEnv, SATProblem, linear_beta_schedule
from data.generate_instances import generate_uniform_kcnf, generate_scalefree_kcnf


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _device_str(device: str) -> str:
    """Resolve 'auto' into a SB3-compatible device string."""
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_vecenv_mode(n_envs: int) -> str:
    """
    Heuristic:
    - On macOS, SubprocVecEnv often costs a lot (spawn/forkserver), so default to DummyVecEnv.
    - Otherwise, use DummyVecEnv for n_envs==1, and SubprocVecEnv for n_envs>1.
    """
    if sys.platform == "darwin":
        return "dummy"
    return "dummy" if n_envs <= 1 else "subproc"


def _default_start_method() -> str:
    """
    Choose a reasonable default start method.
    - On macOS and Windows, spawn is the safest.
    - On Linux, fork is fast but can be unsafe with some libraries; keep spawn as default-safe.
    You can override via CLI.
    """
    if sys.platform in ("darwin", "win32"):
        return "spawn"
    return "spawn"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO policy for RLNMC (MultiBinary backbone mask).")

    # Problem / instance generation
    p.add_argument("--problem_type", choices=["uniform", "scalefree"], required=True)
    p.add_argument("--n_vars", type=int, default=250)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--clause_ratio", type=float, default=9.2)
    p.add_argument("--alpha", type=float, default=2.6, help="Scale-free power-law exponent (used if scalefree)")
    p.add_argument("--b", type=float, default=None, help="Alternative scale-free parameter b (overrides alpha)")
    p.add_argument("--planted", action="store_true", help="Generate planted satisfiable instances")

    # RLNMC / Env parameters
    p.add_argument("--episode_nmc_steps", type=int, default=50, help="# of NMC jumps per episode")
    p.add_argument("--nmc_cycles", type=int, default=3, help="# of cycles inside one NMC jump")
    p.add_argument("--nmc_sweeps", type=int, default=200, help="Sweeps used inside NMC final stage")

    p.add_argument(
        "--include_assignment_in_obs",
        action="store_true",
        help="If set, include the raw assignment x in the observation (MUST match inference config).",
    )

    # SA schedule params for warm-up + background sampling
    p.add_argument("--beta_i", type=float, default=2.0)
    p.add_argument("--beta_nmc", type=float, default=5.0)
    p.add_argument("--beta_f", type=float, default=8.0)
    p.add_argument("--sa_warmup_sweeps", type=int, default=3000, help="SA sweeps before RL starts")

    # Vectorized training
    p.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments (>=1)")

    # Speed knobs (NO change in algo quality)
    p.add_argument(
        "--vec_env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
        help="Vectorized env backend. 'auto' defaults to DummyVecEnv on macOS to reduce overhead.",
    )
    p.add_argument(
        "--subproc_start_method",
        choices=["spawn", "fork", "forkserver"],
        default=None,
        help="Start method for SubprocVecEnv. If not set, a safe default is chosen.",
    )
    p.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="If >0, set torch.set_num_threads(torch_threads) to control CPU threading.",
    )
    p.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable tensorboard logging (reduces overhead in smoke runs).",
    )
    p.add_argument("--verbose", type=int, default=1, help="SB3 verbosity (0/1/2).")

    # PPO training
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging / checkpoints
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save_dir", type=str, default="training/checkpoints")
    p.add_argument("--log_dir", type=str, default="training/tb")
    p.add_argument("--checkpoint_freq", type=int, default=50_000)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    save_dir = _mkdir(args.save_dir)
    log_dir = _mkdir(args.log_dir)

    # Global seeding for reproducibility
    set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Torch CPU threading control (speed knob)
    if torch is not None and int(args.torch_threads) > 0:
        try:
            torch.set_num_threads(int(args.torch_threads))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Build a beta schedule consistent with env/sat_env.RLNMCEnv
    # ------------------------------------------------------------------
    n_warmup = int(args.sa_warmup_sweeps)
    n_episode_steps = int(args.episode_nmc_steps)
    if n_warmup < 0 or n_episode_steps <= 0:
        raise ValueError("sa_warmup_sweeps must be >= 0 and episode_nmc_steps must be > 0")

    beta_parts = []
    if n_warmup > 0:
        beta_parts.append(linear_beta_schedule(args.beta_i, args.beta_nmc, n_warmup))
    beta_parts.append(linear_beta_schedule(args.beta_nmc, args.beta_f, n_episode_steps))
    beta_schedule = np.concatenate(beta_parts).astype(np.float32)

    nmc_beta_start_idx = int(n_warmup)
    n_envs = int(args.n_envs)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------
    def make_env_fn(rank: int):
        def _init():
            rng = np.random.default_rng(int(args.seed) + int(rank))
            n_clauses = int(round(float(args.clause_ratio) * int(args.n_vars)))

            if args.problem_type == "uniform":
                inst = generate_uniform_kcnf(
                    n_vars=args.n_vars,
                    n_clauses=n_clauses,
                    k=args.k,
                    planted=bool(args.planted),
                    rng=rng,
                    avoid_duplicates=True,
                )
            else:
                inst = generate_scalefree_kcnf(
                    n_vars=args.n_vars,
                    n_clauses=n_clauses,
                    k=args.k,
                    planted=bool(args.planted),
                    rng=rng,
                    alpha=float(args.alpha),
                    b=args.b,
                    avoid_duplicates=True,
                )

            problem = SATProblem(n_vars=int(inst.n_vars), clauses=inst.clauses)

            env = RLNMCEnv(
                problem=problem,
                beta_schedule=beta_schedule,
                nmc_beta_start_idx=nmc_beta_start_idx,
                n_cycles=int(args.nmc_cycles),
                n_relax_sweeps=1,
                n_full_sweeps=int(max(1, args.nmc_sweeps)),
                seed=int(args.seed) + int(rank),
                terminate_on_solution=True,
                include_assignment_in_obs=bool(args.include_assignment_in_obs),
            )
            return env

        return _init

    env_fns = [make_env_fn(i) for i in range(n_envs)]

    # Decide VecEnv backend
    vec_mode = str(args.vec_env)
    if vec_mode == "auto":
        vec_mode = _default_vecenv_mode(n_envs)

    if vec_mode == "dummy" or n_envs <= 1:
        vec_env = DummyVecEnv(env_fns)
        start_method_used: Optional[str] = None
    else:
        start_method = args.subproc_start_method or _default_start_method()
        vec_env = SubprocVecEnv(env_fns, start_method=str(start_method))
        start_method_used = str(start_method)

    # PPO model
    device = _device_str(args.device)

    tb_log = None if bool(args.no_tensorboard) else log_dir

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=int(args.verbose),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=tb_log,
        device=device,
    )

    # Save run config for reproducibility
    _save_json(
        os.path.join(save_dir, "train_config.json"),
        {
            "args": vars(args),
            "env": {
                "problem_type": args.problem_type,
                "n_vars": args.n_vars,
                "k": args.k,
                "clause_ratio": args.clause_ratio,
                "alpha": args.alpha,
                "b": args.b,
                "planted": bool(args.planted),
                "beta_i": args.beta_i,
                "beta_nmc": args.beta_nmc,
                "beta_f": args.beta_f,
                "sa_warmup_sweeps": int(args.sa_warmup_sweeps),
                "episode_nmc_steps": int(args.episode_nmc_steps),
                "nmc_cycles": int(args.nmc_cycles),
                "nmc_sweeps": int(args.nmc_sweeps),
                "include_assignment_in_obs": bool(args.include_assignment_in_obs),
            },
            "vec_env": {
                "mode": vec_mode,
                "n_envs": n_envs,
                "subproc_start_method": start_method_used,
            },
            "sb3": {
                "algo": "PPO",
                "policy": "MlpPolicy",
                "device": device,
                "tensorboard_log": None if tb_log is None else str(tb_log),
                "verbose": int(args.verbose),
            },
        },
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=int(args.checkpoint_freq),
        save_path=save_dir,
        name_prefix="ppo_rlnmc",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=int(args.total_timesteps), callback=checkpoint_cb)

    final_path = os.path.join(save_dir, "ppo_rlnmc_final.zip")
    model.save(final_path)
    print(f"\nTraining finished. Final model saved to: {final_path}")

    vec_env.close()


if __name__ == "__main__":
    main()