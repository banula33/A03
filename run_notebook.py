"""Execute and save after each cell, preserving results if a later cell fails."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

os.environ["JUPYTER_PATH"] = str(Path(".jupyter/share/jupyter").resolve())
os.environ["IPYTHONDIR"] = str(Path(".jupyter/ipython").resolve())
path = Path("230070T_A03.ipynb")
notebook = nbformat.read(path, as_version=4)
class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        output = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            text = msg["content"]["text"]
            if "Epoch" in text or "Error" in text:
                print(text, end="", flush=True)
        return output

client = ProgressClient(notebook, timeout=None, kernel_name="assignment",
                        resources={"metadata": {"path": str(Path.cwd())}})
with client.setup_kernel():
    for i, cell in enumerate(notebook.cells):
        print(f"Starting cell {i}: {cell.source.splitlines()[0]}", flush=True)
        try:
            client.execute_cell(cell, i)
        finally:
            nbformat.write(notebook, path)
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text[-3000:], flush=True)
        print(f"Finished cell {i}", flush=True)
print("Notebook complete", flush=True)
