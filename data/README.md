Data directory placeholders. DO NOT commit biomedical data.

Person 1 must supply processed EEG data under:

```
data/processed/eeg/
    X.npy
    y.npy
    subject_id.npy
    trial_id.npy
    metadata.csv
    channel_names.json
    label_map.json
    preprocessing_config.json
```

This repository preserves the directory structure using .gitkeep files.
