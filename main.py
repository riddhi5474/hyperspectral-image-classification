import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "CZ_hsdb"  # folder with .mat files
MAX_PIXELS_PER_FILE = 100000  # random sample size per file
N_COMPONENTS_PCA = None  # set to e.g. 10 if you want PCA

# ----------------------------
# Load all .mat files
# ----------------------------
all_X, all_y = [], []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".mat"):
        continue

    print(f"Loading {fname} ...")
    data = sio.loadmat(os.path.join(DATA_DIR, fname))

    cube = data["ref"]  # shape (H, W, B)
    labels = data["lbl"]  # shape (H, W)

    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32)  # (H*W, B)
    y = labels.flatten()

    # remove unlabeled pixels (often 0 or <0)
    mask = y > 0
    X, y = X[mask], y[mask]

    # sample if too large
    if len(y) > MAX_PIXELS_PER_FILE:
        idx = np.random.choice(len(y), size=MAX_PIXELS_PER_FILE, replace=False)
        X, y = X[idx], y[idx]

    all_X.append(X)
    all_y.append(y)

# concatenate across all files
X = np.vstack(all_X)
y = np.hstack(all_y)

print("Final dataset:", X.shape, y.shape)

# ----------------------------
# Optional: PCA
# ----------------------------
if N_COMPONENTS_PCA is not None:
    from sklearn.decomposition import PCA

    print(f"Reducing to {N_COMPONENTS_PCA} PCA components ...")
    pca = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
    X = pca.fit_transform(X)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Train classifier
# ----------------------------
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
