# Embodied squad-commander report

`report.tex` is the source for the related-work review, protocol specification,
implementation map, and preregistered Luna evaluation plan. `report.pdf` is the
compiled pre-experiment artifact. Its pending-result language is retained as a
historical record of what was claimed before evaluation; completed 100-tick
outcomes are reported separately below.

`first_test_preregistration.md` is the historical, review-gated specification for
the proposed 30-step paired test. Its pre-run status is intentionally not
rewritten after later work.

`100_tick_evaluation.md` contains the frozen protocol and the completed manual
audit of the one-seed, 100-tick Source-versus-commander evaluation. The
preregistered gate failed on planner-format validity; the report separates the
positive handover signal from the negative general-progress, reliability, and
cost results.

Rebuild from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

Primary-paper metadata is kept in `references.bib`. Generated LaTeX auxiliaries
are not research artifacts and should remain untracked.
