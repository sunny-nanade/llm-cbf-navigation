"""
Generate a system architecture diagram (DOT and SVG/PDF) for the paper.
Requires: python-graphviz package (installed). For SVG/PDF output, Graphviz 'dot' binary must be installed on the system PATH.
"""
from graphviz import Digraph
import shutil
import os
from pathlib import Path

OUT_DIR = Path('figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

g = Digraph('EmbodiedAI', format='svg')
g.attr(rankdir='LR', fontsize='12', labelloc='t', label='Embodied Robot: Sensing → Fusion → LLM Planning → Adaptive Control → Safety RA → Actuation')

with g.subgraph(name='cluster_sensors') as s:
    s.attr(label='Sensors', style='rounded')
    s.node('cam', 'RGB-D Camera')
    s.node('lidar', 'LiDAR')
    s.node('tact', 'Tactile')
    s.node('prop', 'Proprioception\n(IMU/Joint States)')

with g.subgraph(name='cluster_fusion') as f:
    f.attr(label='Fusion Layer', style='rounded')
    f.node('sync', 'Sync + Calibration')
    f.node('est', 'State Estimation\n(UKF / Factor Graph)')

with g.subgraph(name='cluster_llm') as l:
    l.attr(label='LLM Planner', style='rounded')
    l.node('ctx', 'Context Encoding\n(scene summary)')
    l.node('plan', 'Constrained Prompting\n+ Confidence Gating')

with g.subgraph(name='cluster_ctrl') as c:
    c.attr(label='Adaptive Control', style='rounded')
    c.node('ctrl', 'PID/Impedance or MPC\n(online gain tuning)')

with g.subgraph(name='cluster_ra') as r:
    r.attr(label='Runtime Assurance (Safety)', style='rounded')
    r.node('mon', 'Monitors\n(invariants, innovation, collision)')
    r.node('filter', 'Safety Filter\n(project to safe set)')

with g.subgraph(name='cluster_act') as a:
    a.attr(label='Actuators', style='rounded')
    a.node('base', 'Base Velocity')
    a.node('arm', 'Arm Joints')

# Edges
for sensor in ['cam','lidar','tact','prop']:
    g.edge(sensor, 'sync')

g.edge('sync', 'est')
g.edge('est', 'ctx')
g.edge('ctx', 'plan')
g.edge('plan', 'ctrl', label='nominal cmd')
g.edge('ctrl', 'filter', label='proposed cmd')
g.edge('est', 'mon', label='state/confidence')
g.edge('mon', 'filter', label='constraints')
g.edge('filter', 'base')
g.edge('filter', 'arm')

# Save outputs
DOT_PATH = OUT_DIR / 'architecture.dot'
SVG_PATH = OUT_DIR / 'architecture.svg'
PDF_PATH = OUT_DIR / 'architecture.pdf'

DOT_PATH.write_text(g.source, encoding='utf-8')
print(f'[OK] Wrote DOT: {DOT_PATH.resolve()}')

# Try to locate Graphviz dot if not on PATH (Windows common install path)
dot_ok = shutil.which('dot') is not None
if not dot_ok:
    common_dot = r"C:\\Program Files\\Graphviz\\bin\\dot.exe"
    if Path(common_dot).exists():
        os.environ['PATH'] = str(Path(common_dot).parent) + os.pathsep + os.environ.get('PATH', '')
        dot_ok = shutil.which('dot') is not None

if dot_ok:
    # Render SVG
    g.format = 'svg'
    g.render(filename='architecture', directory=str(OUT_DIR), cleanup=True)
    print(f'[OK] Wrote SVG: {SVG_PATH.resolve()}')
    # Render PDF
    g.format = 'pdf'
    g.render(filename='architecture', directory=str(OUT_DIR), cleanup=True)
    print(f'[OK] Wrote PDF: {PDF_PATH.resolve()}')
else:
    print('[WARN] Graphviz dot not found on PATH. Install Graphviz to render SVG, or use the .dot file.')
