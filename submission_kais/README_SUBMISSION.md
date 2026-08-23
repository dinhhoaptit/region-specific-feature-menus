# KAIS upload package

Rebuild (this folder):

```text
pdflatex Region-Specific_IFM_for_PFR
bibtex   Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
```

## Upload to the journal (do not include `submission_kbs/`)

### Main manuscript

| File | Role |
|---|---|
| `Region-Specific_IFM_for_PFR.pdf` | Compiled article |
| `Region-Specific_IFM_for_PFR_latex_sources.zip` | TeX, bibliography, class files, figure PDFs |
| `COVER_LETTER.md` | Cover letter (paste or convert to PDF in the submission system) |

### Electronic supplementary material

| File | Role |
|---|---|
| `Region-Specific_IFM_for_PFR_code.zip` | Python source to reproduce tables and figures |
| `Region-Specific_IFM_for_PFR_supplement.zip` | JSON numeric records for the reported tables |

Do **not** upload `alternating_*.json` (partition-refinement trials that are not in the paper).

Do **not** upload local `data/*.npz` (public UCI/OpenML; download script is in the code zip).
