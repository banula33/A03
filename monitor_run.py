"""Show progress while the notebook kernel runs a long cell."""
import time
from pathlib import Path

for _ in range(120):
    log = Path("execution.log").read_text()
    print(log[-1500:], flush=True)
    for path in Path("data").rglob("*"):
        if path.is_file() and (path.suffix in (".gz", ".zip") or "resnet18" in path.name):
            print(path, round(path.stat().st_size / 1024**2, 1), "MB", flush=True)
    if "Notebook complete" in log or "Traceback (most recent call last)" in log:
        break
    time.sleep(45)
