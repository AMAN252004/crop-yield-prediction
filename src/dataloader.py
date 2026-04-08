# src/dataloader.py
"""
Dataloader utilities for the processed CNN-RNN dataset.

Expecting files:
  data/processed/Xw.npy  -> shape (N, 52, 6)
  data/processed/Xs.npy  -> shape (N, 66)
  data/processed/Xm.npy  -> shape (N, 14)
  data/processed/y.npy   -> shape (N,)

Functions:
  load_processed()           -> loads arrays and meta
  train_val_test_split()     -> returns train/val/test numpy splits and saves indices
  get_tf_datasets()          -> converts numpy splits into tf.data.Datasets
  add_gdd_and_rain_anom()    -> simple hooks (not used by default)
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"
os.makedirs(SPLITS_DIR, exist_ok=True)

def load_processed(processed_dir=PROCESSED_DIR):
    """Load processed .npy arrays and meta.json"""
    Xw = np.load(os.path.join(processed_dir, "Xw.npy"))
    Xs = np.load(os.path.join(processed_dir, "Xs.npy"))
    Xm = np.load(os.path.join(processed_dir, "Xm.npy"))
    y  = np.load(os.path.join(processed_dir, "y.npy"))
    meta_path = os.path.join(processed_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    print(f"Loaded processed arrays from {processed_dir}:")
    print("  Xw:", Xw.shape)
    print("  Xs:", Xs.shape)
    print("  Xm:", Xm.shape)
    print("  y :", y.shape)
    return Xw, Xs, Xm, y, meta

def simple_impute(arr):
    """Replace NaN with column mean (in-place safe copy)"""
    a = np.array(arr, dtype=float)
    if np.isnan(a).any():
        colmean = np.nanmean(a, axis=0)
        inds = np.where(np.isnan(a))
        a[inds] = np.take(colmean, inds[1])
    return a

def train_val_test_split(Xw, Xs, Xm, y,
                         test_size=0.2, val_size=0.1,
                         random_state=42, stratify=None,
                         save_splits=True):
    """
    Split into train, val, test sets.
      - test_size: fraction for test
      - val_size: fraction of remaining for validation
    Returns dictionaries with numpy arrays.
    """
    N = len(y)
    idx = np.arange(N)
    # first split test
    idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)
    # then split train/val
    val_relative = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_relative, random_state=random_state, stratify=(y[idx_trainval] if stratify is not None else None))

    def slice_idx(i):
        return {
            "Xw": simple_impute(Xw[i]),
            "Xs": simple_impute(Xs[i]) if Xs is not None else None,
            "Xm": simple_impute(Xm[i]) if Xm is not None else None,
            "y" : y[i]
        }

    train = slice_idx(idx_train)
    val   = slice_idx(idx_val)
    test  = slice_idx(idx_test)

    # Save indices for reproducibility
    if save_splits:
        np.save(os.path.join(SPLITS_DIR, "idx_train.npy"), idx_train)
        np.save(os.path.join(SPLITS_DIR, "idx_val.npy"), idx_val)
        np.save(os.path.join(SPLITS_DIR, "idx_test.npy"), idx_test)
        print(f"Saved split indices to {SPLITS_DIR}")

    print("Split sizes (train/val/test):", len(idx_train), len(idx_val), len(idx_test))
    return train, val, test

def fit_scalers(train):
    """
    Fit StandardScaler to Xs (soil) and Xm (mgmt) and return scaled arrays + scalers dict.
    Xw is left as-is (CNN input).
    """
    scalers = {}
    Xs_train = train["Xs"]
    Xm_train = train["Xm"]

    if Xs_train is not None:
        sc_soil = StandardScaler().fit(Xs_train)
        scalers['soil'] = sc_soil
    else:
        sc_soil = None

    if Xm_train is not None:
        sc_mgmt = StandardScaler().fit(Xm_train)
        scalers['mgmt'] = sc_mgmt
    else:
        sc_mgmt = None

    return scalers

def apply_scalers(arr_dict, scalers):
    """
    Apply fitted scalers to a dictionary with keys 'Xs' and 'Xm'.
    Returns new dict with scaled Xs, Xm.
    """
    out = {}
    out['Xw'] = arr_dict['Xw']
    if arr_dict['Xs'] is not None and scalers.get('soil') is not None:
        out['Xs'] = scalers['soil'].transform(arr_dict['Xs'])
    else:
        out['Xs'] = arr_dict['Xs']
    if arr_dict['Xm'] is not None and scalers.get('mgmt') is not None:
        out['Xm'] = scalers['mgmt'].transform(arr_dict['Xm'])
    else:
        out['Xm'] = arr_dict['Xm']
    out['y'] = arr_dict['y']
    return out

def _pack_inputs(Xw, Xs, Xm):
    """
    Pack inputs in the same order expected by our models:
    [Xw, Xs, Xm] (Xm optional)
    """
    inputs = [Xw]
    if Xs is not None:
        inputs.append(Xs)
    if Xm is not None:
        inputs.append(Xm)
    return inputs

def get_tf_datasets(train, val, test, scalers=None, batch_size=32, shuffle=True):
    """
    Returns tf.data.Dataset objects for train/val/test.
    """

    train_s = apply_scalers(train, scalers) if scalers else train
    val_s   = apply_scalers(val, scalers) if scalers else val
    test_s  = apply_scalers(test, scalers) if scalers else test

    # pack inputs
    Xw_tr, Xs_tr, Xm_tr, y_tr = train_s['Xw'], train_s['Xs'], train_s['Xm'], train_s['y']
    Xw_va, Xs_va, Xm_va, y_va = val_s['Xw'], val_s['Xs'], val_s['Xm'], val_s['y']
    Xw_te, Xs_te, Xm_te, y_te = test_s['Xw'], test_s['Xs'], test_s['Xm'], test_s['y']

    # convert to tf.data
    def to_tf_dataset(Xw, Xs, Xm, y, batch_size=batch_size, shuffle=shuffle):
        # each sample input will be tuple/list matching model.inputs
        def generator():
            for i in range(len(y)):
                inputs = _pack_inputs(Xw[i:i+1].squeeze(axis=0) if Xw is not None else None,
                                      Xs[i:i+1].squeeze(axis=0) if Xs is not None else None,
                                      Xm[i:i+1].squeeze(axis=0) if Xm is not None else None)
                # generator should return inputs as list + label
                yield inputs, y[i]
        # Build TF dataset via from_tensor_slices for speed if arrays are regular
        # but because Keras handles multiple input arrays, we use from_tensor_slices with tuple(inputs)
        inputs_tuple = tuple(x for x in [Xw, Xs, Xm] if x is not None)
        if inputs_tuple:
            ds = tf.data.Dataset.from_tensor_slices((inputs_tuple, y))
        else:
            ds = tf.data.Dataset.from_tensor_slices((y, y))  # should not happen
        if shuffle:
            ds = ds.shuffle(buffer_size=min(10000, len(y)), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = to_tf_dataset(Xw_tr, Xs_tr, Xm_tr, y_tr, batch_size=batch_size, shuffle=shuffle)
    ds_val   = to_tf_dataset(Xw_va, Xs_va, Xm_va, y_va, batch_size=batch_size, shuffle=False)
    ds_test  = to_tf_dataset(Xw_te, Xs_te, Xm_te, y_te, batch_size=batch_size, shuffle=False)

    print("tf.data datasets created: train/val/test sizes:", len(y_tr), len(y_va), len(y_te))
    return ds_train, ds_val, ds_test, (train_s, val_s, test_s)

# ---------- Simple feature-engineering hooks ----------
def add_gdd_from_temps(Xw, tmax_var_index=0, tmin_var_index=1, base_temp=10.0):
    """
    Compute simple weekly GDD from Xw array.
    Xw expected shape: (N, weeks, vars)
    tmax_var_index, tmin_var_index: indices of tmax/tmin among vars (0-based)
    Returns gdd array shape (N, weeks, 1)
    """
    tmax = Xw[..., tmax_var_index]
    tmin = Xw[..., tmin_var_index]
    avg = (tmax + tmin) / 2.0
    gdd = np.maximum(0.0, avg - base_temp)[..., np.newaxis]
    return gdd

def add_weekly_rain_anomaly(Xw, precip_var_index=2):
    """
    Simple weekly precipitation anomaly: (weekly precip - mean_precip_over_weeks) / std
    Returns anomaly shape (N, weeks, 1)
    """
    precip = Xw[..., precip_var_index]  # (N, weeks)
    mean = np.mean(precip, axis=1, keepdims=True)  # per sample
    std  = np.std(precip, axis=1, keepdims=True) + 1e-8
    anom = ((precip - mean) / std)[..., np.newaxis]
    return anom

# ---------- Example quick runner --------------
if __name__ == "__main__":
    Xw, Xs, Xm, y, meta = load_processed()
    train, val, test = train_val_test_split(Xw, Xs, Xm, y, test_size=0.2, val_size=0.1, random_state=42)
    scalers = fit_scalers(train)
    ds_train, ds_val, ds_test, splits = get_tf_datasets(train, val, test, scalers=scalers, batch_size=32)
    # quick sanity: inspect one batch
    for batch in ds_train.take(1):
        (inputs_batch, labels_batch) = batch
        print("One batch inputs types:", type(inputs_batch), isinstance(inputs_batch, tuple))
        print("Labels batch shape:", labels_batch.shape)
    print("Done.")