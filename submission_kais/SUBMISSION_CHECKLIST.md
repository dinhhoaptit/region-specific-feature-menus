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
- Comparative study: global stepwise, localized Lasso–LARS, VIF, abs-corr, random, LRG; nested pre-test for when to regionalize

## Files in this folder

| File | Role |
|---|---|
| `COVER_LETTER.md` | Cover letter to KAIS |
| `sn-jnl.cls`, `sn-basic.bst` | Springer Nature class and numbered bibliography style |
| `Region-Specific_IFM_for_PFR.pdf` | Compiled manuscript (after build) |
| `references.bib` | Bibliography |
| `figures/` | Figure assets |
| `when_to_regionalize.json` | Nested global-vs-region pre-test (Job 3) |
| `named_menus.json` | Named UCI prefixes for Table tab:named |
| `CODE_AVAILABILITY.md` | How to run the code |
| `README_SUBMISSION.md` | What to upload |
| `Region-Specific_IFM_for_PFR_latex_sources.zip` | LaTeX source package (built) |
| `Region-Specific_IFM_for_PFR_code.zip` | Code supplement (built) |
| `Region-Specific_IFM_for_PFR_supplement.zip` | JSON table records (built) |

Do not upload `submission_kbs/` as part of this submission.

## Springer packaging notes

- Springer Nature LaTeX class `sn-jnl.cls` with bibliography style `sn-basic.bst` (numbered citations in square brackets).
- Title page: title, corresponding author (star), email, affiliation (institution, city, country), ORCID.
- Abstract 150--250 words; 6 indexing keywords; acknowledgments on the title page.
- Heading ``Statements and Declarations'' with Competing interests (required), funding, author contributions, ethics/consent (not applicable), data and code availability.
- Decimal headings, at most three levels; footnotes not endnotes; figure captions use bold ``Fig.'' without a closing period.
- Upload `sn-jnl.cls`, `sn-basic.bst`, `Region-Specific_IFM_for_PFR.tex`, `references.bib`, figures, and the compiled PDF.
- KAIS does not require Elsevier Highlights. Do not upload `submission_kbs/HIGHLIGHTS.md`.

## Compile

```text
pdflatex Region-Specific_IFM_for_PFR
bibtex   Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
pdflatex Region-Specific_IFM_for_PFR
```

## Remaining author action for code release

The implementation is complete in this project. There is not yet a public GitHub URL. Create a public repository from the project root (`src/`, `examples/`, `requirements.txt`, `README.md`, `data/README.md`) and insert the URL in the Data and code availability paragraph before upload.
