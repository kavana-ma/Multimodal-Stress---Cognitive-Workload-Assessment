import scipy.io as sio
import os

path = os.path.join("dataset", "VP001-NIRS", "cnt_nback.mat")

data = sio.loadmat(path)
cnt = data["cnt_nback"]

print("cnt_nback shape:", cnt.shape)
print("dtype:", cnt.dtype)

oxy = cnt["oxy"][0, 0]
deoxy = cnt["deoxy"][0, 0]

print("\nOXY")
print("Type:", type(oxy))
print("Shape:", oxy.shape)
print("Dtype:", oxy.dtype)

print("\nDEOXY")
print("Type:", type(deoxy))
print("Shape:", deoxy.shape)
print("Dtype:", deoxy.dtype)

print("\nOXY first few values:")
print(oxy[:5])

print("\nDEOXY first few values:")
print(deoxy[:5])