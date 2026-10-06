import scipy.io as sio
import os
import numpy as np

path = os.path.join(
    "dataset",
    "VP001-NIRS",
    "mrk_nback.mat"
)

data = sio.loadmat(path)
mrk = data["mrk_nback"][0, 0]

times_ms = mrk["time"].flatten()
y = mrk["y"]

class_names = [
    "0-back",
    "2-back",
    "3-back"
]

print("Number of markers:", len(times_ms))
print()

for i, time_ms in enumerate(times_ms):

    active_class = np.argmax(y[:, i])

    print(
        f"{i+1:2d}. "
        f"{class_names[active_class]:5s}  "
        f"time = {time_ms:8d} ms  "
        f"({time_ms / 1000:8.3f} s)"
    )

print("\nClass counts:")

for class_id, name in enumerate(class_names):
    count = np.sum(y[class_id] == 1)
    print(f"{name}: {count}")

print("\nTime differences between consecutive markers:")

diffs = np.diff(times_ms) / 1000

for i, diff in enumerate(diffs):
    print(
        f"{i+1:2d} -> {i+2:2d}: "
        f"{diff:.3f} seconds"
    )

print("\nSampling rate:")
print("10 Hz")

print("\nTotal recording duration:")
print("21178 / 10 =", 21178 / 10, "seconds")