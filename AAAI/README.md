# AlemDICE AAAI-27 draft

This directory contains an anonymous AAAI-27 working draft. The title,
abstract, methods, results, and limitations reflect the current evidence.

The current claim boundary is intentional: the Contract Net/CU implementation
has been mechanically validated, but no corrected matched LLM comparison has
yet established a behavioral coordination benefit.

The artifact root README describes the retained code and reproduction order.
Experiment methods and evidence rules are in `docs/EXPERIMENTS.md` and
`EVALUATION.md`; curated inputs to manuscript figures are in `Results/`.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex
```

The style, bibliography, and checklist content is vendored from the official
[AAAI-27 Author Kit](https://aaai.org/authorkit27/), template version 2027.1.
The reproducibility checklist remains unfilled and should be completed only
after the experimental design and reported results stabilize.
