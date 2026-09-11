"""Build manuscript PDF and KAIS upload zips (run from project root)."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAIS = ROOT / "submission_kais"
STEM = "Region-Specific_IFM_for_PFR"


def compile_pdf() -> None:
    for ext in (".aux", ".bbl", ".blg", ".log", ".out"):
        p = KAIS / f"{STEM}{ext}"
        if p.exists():
            p.unlink()
    cmds = [
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
        ["bibtex", STEM],
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=KAIS, check=True)


def zip_latex() -> Path:
    out = KAIS / f"{STEM}_latex_sources.zip"
    files = [
        KAIS / f"{STEM}.tex",
        KAIS / "references.bib",
        KAIS / "sn-jnl.cls",
        KAIS / "sn-basic.bst",
    ]
    figdir = KAIS / "figures"
    figs = [
        p
        for p in figdir.iterdir()
        if p.is_file() and p.name.startswith("fig") and p.suffix.lower() in {".pdf", ".png", ".eps"}
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.name)
        for p in figs:
            zf.write(p, f"figures/{p.name}")
    return out


def zip_code() -> Path:
    out = KAIS / "ESM_5.zip"
    skip_names = {
        "kais_alternating_partition.py",
        "package_kais_submission.py",
        "write_kais_esm.py",
    }
    header = (
        "Article: Region-Specific Incremental Feature Menus for Progressive Feature Revelation\n"
        "Journal: Knowledge and Information Systems\n"
        "Author: Hoa Dinh Nguyen\n"
        "Affiliation: Posts and Telecommunications Institute of Technology, Hanoi, Vietnam\n"
        "Corresponding email: hoand@ptit.edu.vn\n"
        "Online Resource 5 (ESM_5.zip)\n"
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ESM_HEADER.txt", header)
        zf.write(ROOT / "requirements.txt", "requirements.txt")
        zf.write(ROOT / "README.md", "README.md")
        zf.write(ROOT / "data" / "README.md", "data/README.md")
        zf.write(KAIS / "CODE_AVAILABILITY.md", "CODE_AVAILABILITY.md")
        for d in ("src", "examples"):
            base = ROOT / d
            for p in base.rglob("*"):
                if p.is_dir() or p.suffix == ".pyc" or "__pycache__" in p.parts:
                    continue
                if p.name in skip_names or p.name.startswith("_probe"):
                    continue
                zf.write(p, p.relative_to(ROOT).as_posix())
    return out


def zip_supplement() -> Path:
    out = KAIS / f"{STEM}_supplement.zip"
    names = [
        "extra_evidence.json",
        "named_menus.json",
        "conditional_greedy.json",
        "when_to_regionalize.json",
        "stability_and_budget.json",
        "CODE_AVAILABILITY.md",
        "README_SUBMISSION.md",
    ]
    readme = (
        "Electronic supplementary material for\n"
        "Region-Specific Incremental Feature Menus for Progressive Feature Revelation.\n\n"
        "JSON files record the numbers behind the manuscript tables.\n"
        "Python source is in the companion code zip; public data via examples/download_datasets.py.\n"
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_SUPPLEMENT.txt", readme)
        for name in names:
            p = KAIS / name
            if p.exists():
                zf.write(p, name)
    return out


def main() -> None:
    compile_pdf()
    pdf = KAIS / f"{STEM}.pdf"
    if not pdf.exists():
        raise SystemExit("PDF was not produced")
    latex = zip_latex()
    code = zip_code()
    supp = zip_supplement()
    print(f"pdf    {pdf} ({pdf.stat().st_size} bytes)")
    print(f"latex  {latex} ({latex.stat().st_size} bytes)")
    print(f"code   {code} ({code.stat().st_size} bytes)")
    print(f"suppl  {supp} ({supp.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
