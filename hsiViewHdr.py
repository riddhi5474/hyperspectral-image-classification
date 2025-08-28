import spectral as spy
import matplotlib.pyplot as plt

# Load image
hdr_path = r"D:\hyperspectral images\datasets\Crab pond\Crab pond.hdr"
raw_path = r"D:\hyperspectral images\datasets\Crab pond\Crab pond.raw"
img = spy.envi.open(hdr_path, raw_path)

# Read data into array
cube = img.load()  # shape = (lines, samples, bands)

# Select RGB bands [R,G,B]
r, g, b = 77, 44, 26
rgb = cube[:, :, [r, g, b]]

# Normalize 0-1
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(10, 8))
plt.imshow(rgb)
plt.title("Crab Pond - False Color RGB (660, 550, 490 nm approx.)")
plt.axis("off")
plt.show()

plt.imshow(cube[:, :, 44], cmap="gray")  # Band ~550nm (green)
plt.title("Band 44 (~550 nm)")
plt.axis("off")
plt.show()

row, col = 1000, 1500  # pixel coordinates
spectrum = cube[row, col, :].ravel()  # flatten to (176,)

plt.plot(spectrum)
plt.title(f"Spectral Signature at pixel ({row}, {col})")
plt.xlabel("Band Index (1–176)")
plt.ylabel("Reflectance / DN")
plt.grid(True)
plt.show()
