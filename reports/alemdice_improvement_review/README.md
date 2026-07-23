# AlemDICE improvement review

This folder contains a source-backed technical review of ways to improve the
AlemDICE LLM-agent baseline while preserving the published Alem control.

- `report.tex` — LaTeX source
- `references.bib` — bibliography
- `report.pdf` — compiled report

Rebuild with:

```bash
cd reports/alemdice_improvement_review
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

The review was prepared against the repository state merged into `main` on
2026-07-21. Recommendations are proposals, not changes to the canonical Alem
evaluation protocol.
