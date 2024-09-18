import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('./NormalMapImages/Top.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./NormalMapImages/Bottom.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('./NormalMapImages/Left.png', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('./NormalMapImages/Right.png', cv2.IMREAD_GRAYSCALE)

# Stack the images into a 3D array
I = np.stack([image1, image2, image3, image4], axis=-1)  # Shape: (height, width, 4)

# Define light source directions for the 4 images
# Assuming: top, bottom, left, right
light_dirs = np.array([
[0, 1, 1],   # Top
[0, -1, 1],  # Bottom
[-1, 0, 1],  # Left
[1, 0, 1]    # Right
])
light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1)[:, np.newaxis]  # Normalize vectors

# Compute surface normals
h, w, n = I.shape
I_reshaped = I.reshape(-1, n)  # Flatten to (h*w, 4)
normals = np.linalg.lstsq(light_dirs, I_reshaped.T, rcond=None)[0].T  # Solve for normals

# Reshape normals back to image size (h, w, 3)
normals = normals.reshape(h, w, 3)

# Normalize normals for visualization
normals_vis = (normals - normals.min()) / (normals.max() - normals.min())

plt.imshow(normals_vis)
plt.title('Surface Normals')
plt.show()