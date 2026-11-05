import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
print("Upload an image:")
uploaded = files.upload()

# Read image
filename = list(uploaded.keys())[0]
img = cv2.imread(filename, 0)

if img is None:
    print("Unable to read image")
else:
    # Thresholding
    ret, thresh = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY
    )

    # Kernel
    kernel = np.ones((5, 5), np.uint8)

    # Opening
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel
    )

    # Closing
    closing = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel
    )

    # Dilation
    dilation = cv2.dilate(
        closing, kernel, iterations=2
    )

    # Display results
    titles = ['Original Image', 'Thresholding',
              'Opening', 'Closing', 'Dilation']

    images = [img, thresh, opening, closing, dilation]

    plt.figure(figsize=(12, 8))

    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Image segmentation completed!")