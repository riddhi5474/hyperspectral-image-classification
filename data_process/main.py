import spectral
import numpy as np
from preprocess import preprocess_image  # your function

# Load HDR hyperspectral image
img = spectral.open_image("D:\hyperspectral images\datasets\Crab pond\Crab pond.hdr")
X = img.load().astype("float32")  # (H, W, bands)

# Reorder to (bands, H, W)
X = np.transpose(X, (2, 0, 1))

# Preprocess into patches
patches = preprocess_image(
    X, pca_components=30, window_size=28, stride=28, augment=True
)

print("Patch shape:", patches.shape)

