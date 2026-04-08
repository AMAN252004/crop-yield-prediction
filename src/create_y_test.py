# src/create_y_test.py

import numpy as np
from src.dataloader import load_processed, train_val_test_split

# Load processed dataset
Xw, Xs, Xm, y, meta = load_processed()

# Use same split logic as other training scripts
train, val, test = train_val_test_split(Xw, Xs, Xm, y, test_size=0.2, val_size=0.1, random_state=42)

# Extract true y_test
y_test = test["y"]

# Save it
np.save("results/y_test.npy", y_test)

print("Saved: results/y_test.npy")
print("Test size:", len(y_test))