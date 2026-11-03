# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from google.colab import files
import io

# Upload image
uploaded = files.upload()

# Read uploaded image
image = Image.open(io.BytesIO(next(iter(uploaded.values()))))

# Resize image
image = image.resize((200, 200))

# Convert image into numpy array
image_array = np.array(image.convert("RGB"))

# Reshape image into 2D array
pixels = image_array.reshape(-1, 3)

# Apply K-Means clustering
k = 3

model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(pixels)

# Get cluster centers
centers = model.cluster_centers_

# Replace pixels with cluster center colors
segmented_image = centers[model.labels_]

# Reshape segmented image
segmented_image = segmented_image.reshape(image_array.shape)

# Convert into integer format
segmented_image = np.uint8(segmented_image)

# Display original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("K-Means Segmented Image")
plt.axis("off")

plt.show()