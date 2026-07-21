# AlemDICE AAAI-27 draft

This directory contains an anonymous AAAI-27 working draft. The manuscript is
deliberately limited to an abstract and informal Methods notes. Visible `TODO`
markers identify decisions or evidence that must be resolved before submission.

The current claim boundary is intentional: the Contract Net/CU implementation
has been mechanically validated, but no corrected matched LLM comparison has
yet established a behavioral coordination benefit.

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

The main source retains two abstract versions. The evidence-focused first
version is commented out, while the vision-led second version is currently
active.
