"""Run a grid of simulation scenarios and aggregate metrics.
Usage:
  python code/sim/run_experiments.py --seeds 5 --steps 400
Outputs:
  - code/sim/out/experiments_summary.csv
  - figures/ablation_metrics.png (comparative bars)
"""
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pybullet_sim import run_sim
from scenarios import get_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    scenarios = get_scenarios()

    rows = []
    for sc in scenarios:
        for seed in range(args.seeds):
            _, metrics = run_sim(steps=args.steps,
                                 latency_steps=sc['latency_steps'],
                                 dropout_prob=sc['dropout_prob'],
                                 noise_std=sc['noise_std'],
                                 seed=seed)
            row = {'scenario': sc['name'], 'seed': seed}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path('code/sim/out/experiments_summary.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote summary: {out_csv.resolve()}")

    # Aggregate and plot
    agg = df.groupby('scenario').agg(
        success_rate=('success', 'mean'),
        steps_to_goal=('steps_to_goal', 'mean'),
        violations=('violations', 'mean'),
        final_goal_dist=('final_goal_dist', 'mean'),
    ).reset_index()

    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context('talk'); sns.set_style('whitegrid')

    plt.figure(figsize=(9,4))
    metrics = ['success_rate', 'steps_to_goal', 'violations']
    for i, m in enumerate(metrics, 1):
        plt.subplot(1,3,i)
        sns.barplot(data=agg, x='scenario', y=m, color='#4C72B0')
        plt.xticks(rotation=30, ha='right')
        plt.title(m.replace('_',' ').title())
        plt.tight_layout()
    out_fig = fig_dir / 'ablation_metrics.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Wrote figure: {out_fig.resolve()}")


if __name__ == '__main__':
    main()
