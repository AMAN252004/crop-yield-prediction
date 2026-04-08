import numpy as np

Xw = np.load("data/processed/Xw.npy")
Xs = np.load("data/processed/Xs.npy")
Xm = np.load("data/processed/Xm.npy")
y  = np.load("data/processed/y.npy")

print("Xw:", Xw.shape)
print("Xs:", Xs.shape)
print("Xm:", Xm.shape)
print("y:", y.shape)