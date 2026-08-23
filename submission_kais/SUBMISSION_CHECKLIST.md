# Knowledge and Information Systems — Submission Checklist

Official pages:
- https://link.springer.com/journal/10115
- https://link.springer.com/journal/10115/aims-and-scope

## Article type

Regular research article (not a Short Paper). KAIS regular papers should stay within about 15,000 words; a full-page figure or table counts as 500 words.

## Journal fit used in this package

- Knowledge discovery / data mining / learning
- Inspectable region-specific feature menus as local acquisition knowledge
- Anytime (budgeted) evaluation plus mixed/negative controls
- Comparative study: global stepwise, localized Lasso–LARS, VIF, abs-corr, random

## Files in this folder

| File | Role |
|---|---|
| `COVER_LETTER.md` | Cover letter to KAIS |
| `Region-Specific_IFM_for_PFR.tex` | Editable LaTeX source |
| `Region-Specific_IFM_for_PFR.pdf` | Compiled manuscript (after build) |
| `references.bib` | Bibliography |
| `figures/` | Figure assets |
| `extra_evidence.json` | Menu-overlap and localized-Lasso numbers |
| `CODE_AVAILABILITY.md` | How to run the code |
| `../src/` and `../examples/` | Implementation (project root) |
| `../README.md` | Reproduction instructions |

Do not upload `submission_kbs/` as part of this submission.

## Springer packaging notes

- Author–year citations (`natbib` + `plainnat`). Production may restyle after acceptance.
- Declarations: author contributions, competing interests, funding, data and code availability.
- KAIS does not require Elsevier Highlights. Do not upload `submission_kbs/HIGHLIGHTS.md`.
- Title page information is on the first page of the manuscript.

## Compile

```text
pdflatex Region-Specific_IFM_for_PFR
bibtex   Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
```

## Remaining author action for code release

The implementation is complete in this project. There is not yet a public GitHub URL. Create a public repository from the project root (`src/`, `examples/`, `requirements.txt`, `README.md`, `data/README.md`) and insert the URL in the Data and code availability paragraph before upload.
