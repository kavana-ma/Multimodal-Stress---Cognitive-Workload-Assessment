# fNIRS Cognitive Workload Assessment

## Dataset

- TU Berlin fNIRS N-back Dataset
- 24 subjects
- 36 channels
- HbO & HbR

## Folder Structure

fnirs_model/
│
├── data/
├── models/
├── results/
├── src/

## Setup

pip install -r requirements.txt

## Run Order

cd src

python load_data.py
python dataset.py
python test_loso.py
python train_loso.py
python feature_extraction.py
python train_mlp.py
python evaluate_mlp.py
