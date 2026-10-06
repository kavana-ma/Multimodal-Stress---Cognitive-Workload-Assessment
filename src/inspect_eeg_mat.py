import scipy.io
from pathlib import Path

file_path = Path("dataset/VP001-EEG/mrk_nback.mat")

data = scipy.io.loadmat(
    file_path,
    struct_as_record=False,
    squeeze_me=True
)

mrk = data["mrk_nback"]

print("=" * 70)
print("N-BACK SESSION DURATIONS")
print("=" * 70)

session_indices = []

for i in range(len(mrk.time)):

    active_classes = []

    for class_index in range(len(mrk.className)):
        if mrk.y[class_index, i] == 1:
            active_classes.append(mrk.className[class_index])

    if active_classes and (
        "0-back session" in active_classes
        or "2-back session" in active_classes
        or "3-back session" in active_classes
    ):
        session_indices.append((i, mrk.time[i], active_classes[0]))


for j, (index, start_time, session_name) in enumerate(session_indices):

    # Find the event immediately before the next session
    if j + 1 < len(session_indices):
        next_index = session_indices[j + 1][0]
        last_event_index = next_index - 1
    else:
        last_event_index = len(mrk.time) - 1

    end_time = mrk.time[last_event_index]

    duration = end_time - start_time

    print(
        f"{session_name:20s} | "
        f"start = {start_time:8d} ms | "
        f"last event = {end_time:8d} ms | "
        f"duration = {duration:6d} ms"
    )