# EN3160 Assignment 3

- `230070T_A03.ipynb`: complete notebook, including saved outputs after execution.
- `230070T_a02.pdf`: PDF export (the assignment specifies the `a02` suffix).
- `results/`: plots and `metrics.json` containing measured results.

Name and GitHub profile must be filled in before submission if still marked as missing.
The index number is taken from the original notebook filename.

Open the notebook in Jupyter or VS Code and select a Python environment with the packages
in `requirements.txt`. Run all cells in order. Internet access is needed for the initial
dataset and model-weight downloads. CPU execution can take a while.

For the local environment prepared here:

```bash
.venv/bin/python run_notebook.py
.venv/bin/python finalize_report.py
.venv/bin/python export_pdf.py
```

`build_notebook.py` recreates the notebook source and clears all outputs; do not run it
after training unless you intend to regenerate and rerun the notebook.
`verify_gradients.py` independently checks the manual gradient formulas against autograd.

The notebook uses final-epoch results, full datasets, and a fixed seed. Hymenoptera
evaluation uses its provided validation split. No independent hymenoptera test result
is claimed. GitHub commits or work history are not manufactured by these scripts.
