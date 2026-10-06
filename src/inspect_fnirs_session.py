import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# Load fNIRS data
# -----------------------------

path = os.path.join(
    "dataset",
    "VP001-NIRS",
    "cnt_nback.mat"
)

data = sio.loadmat(path)
cnt = data["cnt_nback"][0, 0]

oxy = cnt["oxy"][0, 0]
deoxy = cnt["deoxy"][0, 0]

oxy_signal = oxy["x"]
deoxy_signal = deoxy["x"]

fs = float(oxy["fs"][0, 0])

print("Sampling rate:", fs)
print("HbO shape:", oxy_signal.shape)
print("HbR shape:", deoxy_signal.shape)

# -----------------------------
# Load markers
# -----------------------------

marker_path = os.path.join(
    "dataset",
    "VP001-NIRS",
    "mrk_nback.mat"
)

marker_data = sio.loadmat(marker_path)
mrk = marker_data["mrk_nback"][0, 0]

times_ms = mrk["time"].flatten()
y = mrk["y"]

class_names = [
    "0-back",
    "2-back",
    "3-back"
]

# First marker
marker_sample = int(times_ms[0] / 1000 * fs)
marker_class = np.argmax(y[:, 0])

print("First marker:")
print("  Time:", times_ms[0] / 1000, "seconds")
print("  Class:", class_names[marker_class])
print("  Sample:", marker_sample)

# -----------------------------
# Extract 80 seconds
# -----------------------------

start = marker_sample
end = min(
    marker_sample + int(80 * fs),
    oxy_signal.shape[0]
)

segment = oxy_signal[start:end]

time = np.arange(len(segment)) / fs

# -----------------------------
# Plot HbO
# -----------------------------

plt.figure(figsize=(14, 8))

for ch in range(6):
    plt.plot(
        time,
        segment[:, ch],
        label=f"Channel {ch + 1}"
    )

plt.axvline(
    0,
    linestyle="--",
    label="Session marker"
)

plt.xlabel("Time (seconds)")
plt.ylabel("HbO concentration (mmol/L)")
plt.title(
    f"VP001 — First N-back Session ({class_names[marker_class]})"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()