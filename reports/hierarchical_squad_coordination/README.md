# Embodied squad-commander report

`report.tex` is the source for the related-work review, protocol specification,
implementation map, and preregistered Luna evaluation plan. `report.pdf` is the
compiled artifact. Experimental outcomes for the new hierarchy remain pending
until the staged study produces canonical output directories.

`first_test_preregistration.md` is the standalone, review-gated specification for
the first 30-step paired test. It is explicitly marked not run and must be
approved before the Luna preflight or either experimental arm is launched.

`100_tick_evaluation.md` freezes the authorized intermediate evaluation before
provider calls and will be extended with the audited findings after both arms
finish.

Rebuild from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

Primary-paper metadata is kept in `references.bib`. Generated LaTeX auxiliaries
are not research artifacts and should remain untracked.
