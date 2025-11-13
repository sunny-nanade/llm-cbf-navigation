"""Create a clean MDPI submission zip with LaTeX sources, figures, and PDF.
Usage:
  python code/pack_submission.py --out submission/mdpi_submission.zip
"""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = [
    ROOT / 'MDPI_template_ACS',  # main LaTeX source folder
    ROOT / 'figures',            # figures referenced relatively as ../figures
    ROOT / 'refs',               # bibliography
]

EXCLUDE_PATTERNS = {
    '.aux', '.bbl', '.blg', '.out', '.log', '.toc', '.lof', '.lot', '.synctex.gz',
}

def should_include(path: Path) -> bool:
    if path.is_dir():
        return True
    if any(path.name.endswith(ext) for ext in EXCLUDE_PATTERNS):
        return False
    # Include PDFs and generated .tex tables
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='submission/mdpi_submission.zip')
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for base in INCLUDE_DIRS:
            if not base.exists():
                continue
            for p in base.rglob('*'):
                if not should_include(p):
                    continue
                if p.is_file():
                    # Preserve relative layout from repo root
                    arcname = p.relative_to(ROOT)
                    zf.write(p, arcname)

        # Also include the compiled PDF for convenience
        pdf = ROOT / 'MDPI_template_ACS' / 'template.pdf'
        if pdf.exists():
            zf.write(pdf, pdf.relative_to(ROOT))

    print(f"[OK] Wrote submission archive: {out_path.resolve()}")

if __name__ == '__main__':
    main()
