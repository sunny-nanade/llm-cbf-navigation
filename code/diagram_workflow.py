"""
Generate a workflow diagram (DOT -> SVG/PDF) showing the runtime loop.
"""
from graphviz import Digraph
import shutil, os
from pathlib import Path

OUT_DIR = Path('figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

g = Digraph('Workflow', format='svg')
g.attr(rankdir='LR', fontsize='12', labelloc='t', label='Runtime Loop: Sense → Fuse → Plan (LLM) → Control → Safety Filter → Actuate')

g.node('sense', 'Sensors\n(RGB-D, LiDAR, Tactile, Prop)')
g.node('fusion', 'Fusion\n(Sync + State Estimation)')
g.node('plan', 'LLM Planner\n(Constrained + Confidence Gate)')
g.node('ctrl', 'Adaptive Control\n(PID/Impedance/MPC)')
g.node('safety', 'Runtime Assurance\n(CBF QP Projection)')
g.node('act', 'Actuators\n(Base/Arm)')

g.edge('sense', 'fusion')
g.edge('fusion', 'plan', label='context')
g.edge('plan', 'ctrl', label='nominal target')
g.edge('ctrl', 'safety', label='candidate u')
g.edge('safety', 'act', label='safe u')
g.edge('fusion', 'safety', style='dashed', label='invariants')
g.edge('act', 'sense', style='dotted', label='closed-loop')

DOT = OUT_DIR / 'workflow.dot'
SVG = OUT_DIR / 'workflow.svg'
PDF = OUT_DIR / 'workflow.pdf'
DOT.write_text(g.source, encoding='utf-8')
print(f"[OK] Wrote DOT: {DOT.resolve()}")

dot_ok = shutil.which('dot') is not None
if not dot_ok:
    common_dot = r"C:\\Program Files\\Graphviz\\bin\\dot.exe"
    if Path(common_dot).exists():
        os.environ['PATH'] = str(Path(common_dot).parent) + os.pathsep + os.environ.get('PATH', '')
        dot_ok = shutil.which('dot') is not None

if dot_ok:
    g.format = 'svg'
    g.render(filename='workflow', directory=str(OUT_DIR), cleanup=True)
    print(f"[OK] Wrote SVG: {SVG.resolve()}")
    g.format = 'pdf'
    g.render(filename='workflow', directory=str(OUT_DIR), cleanup=True)
    print(f"[OK] Wrote PDF: {PDF.resolve()}")
else:
    print('[WARN] Graphviz dot not found; only DOT written.')
