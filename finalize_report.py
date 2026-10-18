"""Remove download progress from saved outputs and validate the completed notebook."""
import re
import json
from pathlib import Path
import nbformat

path = Path("230070T_A03.ipynb")
notebook = nbformat.read(path, as_version=4)
images = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    assert cell.execution_count is not None, "Unexecuted code cell"
    clean_outputs = []
    for output in cell.outputs:
        assert output.output_type != "error", "Execution error"
        if (output.output_type == "stream" and output.get("name") == "stderr"
                and "Unable to import Axes3D" in output.text):
            continue
        if output.output_type == "stream":
            lines = []
            for line in output.text.splitlines():
                if re.fullmatch(r"\s*\d+(?:\.\d+)?%\s*", line):
                    continue
                if "B/s" in line or "it/s" in line or "Downloading:" in line:
                    continue
                lines.append(line)
            output.text = "\n".join(lines) + "\n"
            if not output.text.strip():
                continue
        if "image/png" in output.get("data", {}):
            images += 1
        clean_outputs.append(output)
    cell.outputs = clean_outputs

note = """### Reading the MNIST results
The small CNN has fewer parameters but takes longer in this run. Parameter count
measures stored weights, while computation also depends on the number of spatial
positions processed by each convolution; batch normalization adds work too.
The displayed errors include 4/9 and 3/5 confusions with overlapping stroke patterns.
These examples help explain some mistakes, but do not explain every test error.
"""
if not any(c.source.startswith("### Reading the MNIST results") for c in notebook.cells):
    position = next(i for i, c in enumerate(notebook.cells)
                    if c.source.startswith("## 3. Transfer learning"))
    notebook.cells.insert(position, nbformat.v4.new_markdown_cell(note))

nbformat.validate(notebook)
assert images >= 12, f"Expected at least 12 figures, got {images}"
results = json.loads(Path("results/metrics.json").read_text())
assert len(results["dense"]["history"]) == 10
for result in list(results["mnist"].values()) + list(results["transfer"].values()):
    assert len(result["history"]) == 10
for result in results["mnist"].values():
    assert sum(map(sum, result["confusion"])) == 10000
for result in results["transfer"].values():
    assert sum(map(sum, result["confusion"])) == 153
fine = results["transfer"]["Fine-tuning"]["history"]
fixed = results["transfer"]["Feature extraction"]["history"]
interpretation = f"""### Interpreting the transfer-learning curves
Fine-tuning reaches {fine[-1]['train_acc']:.2f}% training accuracy, but its final
validation accuracy is {fine[-1]['test_acc']:.2f}%, below its highest observed value
of {max(h['test_acc'] for h in fine):.2f}%. This pattern is consistent with overfitting:
a better fit to the training images does not guarantee better validation accuracy.
Feature extraction ends at {fixed[-1]['test_acc']:.2f}% validation accuracy while
updating far fewer parameters. For this run, keeping the pretrained features fixed
works better at the final epoch. This does not mean fine-tuning is generally worse;
its result depends on the learning rate, training duration and amount of data.
The reported comparison uses epoch 10 for both methods.
"""
if not any(c.source.startswith("### Interpreting the transfer-learning curves") for c in notebook.cells):
    position = next((i for i, c in enumerate(notebook.cells)
                     if c.source.startswith("## References")), len(notebook.cells))
    notebook.cells.insert(position, nbformat.v4.new_markdown_cell(interpretation))
for group in (results["mnist"], results["transfer"]):
    for result in group.values():
        matrix = result["confusion"]
        accuracy = 100 * sum(matrix[i][i] for i in range(len(matrix))) / sum(map(sum, matrix))
        assert abs(accuracy - result["history"][-1]["test_acc"]) < 1e-9
nbformat.write(notebook, path)
print("Validated all code cells, 10-epoch histories, confusion totals and", images, "figures.")
