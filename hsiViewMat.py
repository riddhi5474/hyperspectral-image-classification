import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# load one .mat file
data = sio.loadmat("CZ_hsdb/imge3.mat")  # change to your filename

# check available keys
# print(data.keys())
cube = data['ref']   
labels = data['lbl']  # labels
# print(cube.shape)  
print(labels)
print(labels.shape)
 # e.g. (1040, 1392, 31)

# wavelengths (31 bands, 420–720 nm in 10 nm steps)
wavelengths = np.arange(420, 721, 10)

# Show RGB-like composite (pick 3 bands: R=20, G=10, B=5)
rgb = np.stack([cube[:,:,20], cube[:,:,10], cube[:,:,5]], axis=2)
rgb = rgb / np.max(rgb)   # normalize
plt.imshow(rgb)
plt.title("RGB composite")
plt.show()

# Extract spectrum from a single pixel (row=100, col=200)
spectrum = cube[100,200,:]
plt.plot(wavelengths, spectrum)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Pixel spectrum at (100,200)")
plt.show()

# Average spectrum over a small patch (ROI)
roi = cube[90:110, 190:210, :]     # 20x20 region
mean_spectrum = roi.mean(axis=(0,1))
plt.plot(wavelengths, mean_spectrum, 'r')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Reflectance")
plt.title("Average spectrum (ROI)")
plt.show()