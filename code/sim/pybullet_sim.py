"""
PyBullet simulation with ablations and safety logging.
- Headless (DIRECT) for reproducibility.
- Simple 2D base controller toward a goal with PID and safety clamp.
- Ablations: state-estimation latency, sensor dropout, Gaussian noise, obstacle proximity.
- Outputs per-step CSV and returns aggregate metrics.

CLI example:
  python code/sim/pybullet_sim.py --steps 800 --latency_steps 5 --dropout_prob 0.2 --noise_std 0.02 --scenario latency
"""
import argparse
import json
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data


def run_sim(steps=600,
            goal=(2.0, 2.0),
            kp=1.2, kd=0.3, dt=0.01,
            max_vel=2.0,
            latency_steps=0,
            dropout_prob=0.0,
            noise_std=0.0,
            obstacle=(1.0, 1.0, 0.3),
            seed=0):
    rng = np.random.default_rng(seed)
    OUT_DIR = Path(__file__).parent / 'out'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    planeId = p.loadURDF('plane.urdf')
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF('r2d2.urdf', startPos, startOrientation)

    goal = np.array(goal, dtype=float)
    prev_err = np.zeros(2)
    est_buf = deque(maxlen=max(1, latency_steps + 1))
    pos = np.array([startPos[0], startPos[1]])

    logs = []
    violations = 0
    constraint_acts = 0
    steps_to_goal = None

    for step in range(steps):
        real_pos_xy, _ = p.getBasePositionAndOrientation(robotId)
        real_pos_xy = np.array(real_pos_xy[:2])

        # Sensor dropout or noise applied to measurement
        if dropout_prob > 0 and rng.random() < dropout_prob and len(est_buf) > 0:
            meas = est_buf[-1].copy()  # hold last estimate during dropout
            dropout = 1
        else:
            noise = rng.normal(0, noise_std, size=2) if noise_std > 0 else 0.0
            meas = real_pos_xy + noise
            dropout = 0

        est_buf.append(meas)
        est_pos = est_buf[0] if latency_steps > 0 and len(est_buf) > latency_steps else meas

        err = goal - est_pos
        derr = (err - prev_err)
        v_cmd = kp * err + kd * derr

        pre_speed = np.linalg.norm(v_cmd)
        constraint_active = 1 if pre_speed > max_vel else 0
        if pre_speed > max_vel:
            v_cmd = v_cmd * (max_vel / pre_speed)

        # Safety violation: near obstacle
        ox, oy, orad = obstacle
        dist_obs = np.linalg.norm(real_pos_xy - np.array([ox, oy]))
        violation = 1 if dist_obs < orad else 0

        new_pos = np.append(real_pos_xy + dt * v_cmd, 0.1)
        p.resetBasePositionAndOrientation(robotId, new_pos, startOrientation)

        goal_dist = np.linalg.norm(goal - real_pos_xy)
        if steps_to_goal is None and goal_dist < 0.1:
            steps_to_goal = step

        logs.append({
            'step': step,
            'x': new_pos[0], 'y': new_pos[1],
            'ex': err[0], 'ey': err[1],
            'v_cmd_x': v_cmd[0], 'v_cmd_y': v_cmd[1],
            'speed': np.linalg.norm(v_cmd),
            'goal_dist': goal_dist,
            'constraint_active': constraint_active,
            'violation': violation,
            'dropout': dropout,
            'latency_steps': latency_steps,
            'noise_std': noise_std
        })

        violations += violation
        constraint_acts += constraint_active
        prev_err = err
        p.stepSimulation()

    p.disconnect()

    df = pd.DataFrame(logs)
    success = 1 if steps_to_goal is not None else 0
    if steps_to_goal is None:
        steps_to_goal = steps
    metrics = {
        'success': success,
        'steps_to_goal': steps_to_goal,
        'violations': int(violations),
        'constraint_active_steps': int(constraint_acts),
        'mean_goal_dist': float(df['goal_dist'].mean()),
        'final_goal_dist': float(df['goal_dist'].iloc[-1])
    }
    return df, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--goal_x', type=float, default=2.0)
    ap.add_argument('--goal_y', type=float, default=2.0)
    ap.add_argument('--latency_steps', type=int, default=0)
    ap.add_argument('--dropout_prob', type=float, default=0.0)
    ap.add_argument('--noise_std', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--obstacle_x', type=float, default=1.0)
    ap.add_argument('--obstacle_y', type=float, default=1.0)
    ap.add_argument('--obstacle_r', type=float, default=0.3)
    ap.add_argument('--out', type=str, default='code/sim/out/run_summary.csv')
    args = ap.parse_args()

    df, metrics = run_sim(steps=args.steps,
                          goal=(args.goal_x, args.goal_y),
                          latency_steps=args.latency_steps,
                          dropout_prob=args.dropout_prob,
                          noise_std=args.noise_std,
                          obstacle=(args.obstacle_x, args.obstacle_y, args.obstacle_r),
                          seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(out_path.with_suffix('.metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Wrote simulation log: {out_path.resolve()}")
    print(f"[OK] Metrics: {metrics}")


if __name__ == '__main__':
    main()
