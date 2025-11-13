"""Generate basic plots from simulation CSV.
Usage:
  python code/sim/plot_results.py --csv code/sim/out/run_summary.csv --out figures
Produces trajectory.png, error.png, safety.png in output folder.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--out', type=str, default='figures')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context('talk')
    sns.set_style('whitegrid')

    # Trajectory
    plt.figure(figsize=(6,5))
    plt.plot(df['x'], df['y'], label='trajectory')
    plt.scatter([df['x'].iloc[0]],[df['y'].iloc[0]], c='green', label='start')
    plt.scatter([df['x'].iloc[-1]],[df['y'].iloc[-1]], c='red', label='end')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Agent Trajectory')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'trajectory.png', dpi=150)
    plt.close()

    # Error norm over time
    err_norm = (df['ex']**2 + df['ey']**2)**0.5
    plt.figure(figsize=(6,4))
    plt.plot(df['step'], err_norm)
    plt.xlabel('Step')
    plt.ylabel('Position Error Norm')
    plt.title('Goal Error vs Time')
    plt.tight_layout()
    plt.savefig(out_dir / 'error.png', dpi=150)
    plt.close()

    # Safety signals
    plt.figure(figsize=(6,4))
    plt.plot(df['step'], df['constraint_active'], label='constraint_active')
    plt.plot(df['step'], df['violation'], label='violation')
    plt.xlabel('Step')
    plt.ylabel('Binary Signals')
    plt.title('Safety Constraint & Violations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'safety.png', dpi=150)
    plt.close()

    print(f"[OK] Wrote plots to {out_dir.resolve()}")

if __name__ == '__main__':
    main()