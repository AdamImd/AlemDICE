# Scalable decentralized coordination study

`report.tex` is the source and `report.pdf` is the compiled review, implementation plan, and pilot report. `experiments/registry.yaml` freezes the local pilot before outcomes; `experiments/log.md` is the human audit trail. Raw trajectories remain under `outputs/alem_eval/`, while compact manifests and summaries are copied here for version control.

Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```
