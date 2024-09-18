import numpy as np
import cv2

# Load the four images (grayscale)
image1 = cv2.imread('./NormalMapImages/Right.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./NormalMapImages/Left.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('./NormalMapImages/Above.png', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('./NormalMapImages/Below.png', cv2.IMREAD_GRAYSCALE)

# Stack the images into a single array of shape (H, W, 4)
images = np.stack([image1, image2, image3, image4], axis=-1)

# Define the light direction vectors for each image
light_directions = np.array([
    [1, 0, 1],  # Light direction for image 1
    [-1, 0, 1],  # Light direction for image 2
    [0, 1, 1],  # Light direction for image 3
    [0, -1, 1]   # Light direction for image 4
])

# Invert the light direction matrix
light_directions_inv = np.linalg.pinv(light_directions)

# Initialize normal map
h, w = image1.shape
normals = np.zeros((h, w, 3), dtype=np.float32)

# Compute the normals for each pixel
for y in range(h):
    for x in range(w):
        I = images[y, x, :]  # Intensities for the current pixel
        N = light_directions_inv @ I  # Solve for the normal
        normals[y, x] = N / np.linalg.norm(N)  # Normalize the normal

# Convert normals to a normal map in RGB format
normal_map = (normals + 1) / 2 * 255  # Scale to range [0, 255]
normal_map = normal_map.astype(np.uint8)

# Save or display the normal map
cv2.imwrite('normal_map.png', normal_map)
cv2.imshow('Normal Map', normal_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
