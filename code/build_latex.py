"""build_latex.py - helper script to compile MDPI LaTeX template.
Requires TeX Live (pdflatex, bibtex) installed and on PATH.
Usage (Windows PowerShell):
  python code/build_latex.py --source MDPI_template_ACS --main template.tex
"""
import subprocess, pathlib, argparse, sys

def run(cmd, cwd):
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        # Do not hard-exit on non-zero here; some MiKTeX installs return non-zero despite producing output.
        print(f"[WARN] Command returned non-zero exit code ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='MDPI_template_ACS', help='Folder with template.tex')
    ap.add_argument('--main', default='template.tex', help='Main .tex file name')
    ap.add_argument('--passes', type=int, default=2, help='Extra pdflatex passes after bibtex')
    args = ap.parse_args()

    src = pathlib.Path(args.source).resolve()
    tex = src / args.main
    if not tex.exists():
        print(f"Main TeX file not found: {tex}")
        sys.exit(1)

    # First LaTeX pass
    run(['pdflatex', '-interaction=nonstopmode', tex.name], cwd=src)
    # BibTeX (ignore if .aux missing references)
    aux = src / tex.with_suffix('.aux').name
    if aux.exists():
        run(['bibtex', aux.stem], cwd=src)
    else:
        print('[WARN] AUX file not found after first pass; skipping bibtex.')
    # Additional passes for cross-refs
    for i in range(args.passes):
        run(['pdflatex', '-interaction=nonstopmode', tex.name], cwd=src)

    pdf = src / tex.with_suffix('.pdf').name
    if pdf.exists():
        print(f"[OK] Build complete: {pdf}")
    else:
        print("[ERROR] PDF not generated.")

if __name__ == '__main__':
    main()
