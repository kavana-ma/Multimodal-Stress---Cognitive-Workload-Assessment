import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Load VP001 fNIRS
# ============================================================

subject = "VP001-NIRS"

cnt_path = os.path.join(
    "dataset",
    subject,
    "cnt_nback.mat"
)

data = sio.loadmat(cnt_path)

cnt = data["cnt_nback"][0, 0]

oxy = cnt["oxy"][0, 0]
deoxy = cnt["deoxy"][0, 0]

hbo = oxy["x"]
hbr = deoxy["x"]

fs = float(oxy["fs"][0, 0])

print("Sampling rate:", fs)
print("HbO shape:", hbo.shape)
print("HbR shape:", hbr.shape)


# ============================================================
# Design Butterworth bandpass
# ============================================================

lowcut = 0.01
highcut = 0.20
order = 3

sos = butter(
    order,
    [lowcut, highcut],
    btype="bandpass",
    fs=fs,
    output="sos"
)

print("\nFilter:")
print("Low cutoff :", lowcut, "Hz")
print("High cutoff:", highcut, "Hz")
print("Order      :", order)


# ============================================================
# Filter continuous recording
# ============================================================

hbo_filtered = sosfiltfilt(
    sos,
    hbo,
    axis=0
)

hbr_filtered = sosfiltfilt(
    sos,
    hbr,
    axis=0
)

print("\nFiltered shapes:")
print("HbO:", hbo_filtered.shape)
print("HbR:", hbr_filtered.shape)

print("\nNaN check:")
print("HbO:", np.isnan(hbo_filtered).any())
print("HbR:", np.isnan(hbr_filtered).any())

print("\nInf check:")
print("HbO:", np.isinf(hbo_filtered).any())
print("HbR:", np.isinf(hbr_filtered).any())


# ============================================================
# Inspect first session
# ============================================================

marker_time = 46.560
marker_sample = int(marker_time * fs)

start = marker_sample
end = min(
    marker_sample + int(80 * fs),
    hbo.shape[0]
)

time = np.arange(end - start) / fs

channel = 0

hbo_before = hbo[start:end, channel]
hbo_after = hbo_filtered[start:end, channel]

# ============================================================
# Plot before vs after
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    time,
    hbo_before,
    label="HbO before filtering"
)

plt.plot(
    time,
    hbo_after,
    label="HbO after filtering"
)

plt.axvline(
    0,
    linestyle="--",
    label="Session marker"
)

plt.xlabel("Time (seconds)")
plt.ylabel("HbO concentration (mmol/L)")

plt.title(
    "VP001 — HbO Filtering Test "
    "(Channel 1)"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()