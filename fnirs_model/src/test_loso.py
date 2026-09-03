from loso_split import get_fold
import numpy as np

# Test with Subject 5 as held-out subject
Xtr, ytr, Xte, yte = get_fold(5)

print("="*40)
print("LOSO VERIFICATION")
print("="*40)

print("Train shape :", Xtr.shape)
print("Test shape  :", Xte.shape)

print("\nTrain labels")
print(np.bincount(ytr.astype(int)))

print("\nTest labels")
print(np.bincount(yte.astype(int)))

print("\nTest epochs :", len(yte))