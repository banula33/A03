"""Use the mirror archive only after checking the original CIFAR-10 checksum."""
import hashlib
import tarfile
import time
from pathlib import Path

path = Path("data/cifar-mirror.tar.gz")
print("Waiting for the CIFAR-10 mirror download...", flush=True)
for _ in range(400):
    if path.exists() and path.stat().st_size == 170498071:
        break
    time.sleep(1)
else:
    raise RuntimeError("Mirror download did not finish; original notebook URL is still available.")
checksum = hashlib.md5(path.read_bytes()).hexdigest()
assert checksum == "c58f30108f718f92721af3b95e74349a", checksum
with tarfile.open(path) as archive:
    archive.extractall("data", filter="data")
print("CIFAR-10 original checksum verified and archive extracted.", flush=True)
