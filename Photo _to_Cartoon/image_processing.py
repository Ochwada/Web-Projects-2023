import cv2
import numpy as np
import matplotlib.pyplot as plt

def fetch_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Failed to load image.")
        return None
    else:
        print("Image has been read.")
        return img

def display_image(img, cmap=None):
    if img is None:
        print("Failed to load image.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb, cmap=cmap)
        plt.axis('off')
        plt.show()

def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result 

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def generate_pencil_sketch(img, line_size, blur_value):
    if img is None:
        print("Failed to load image.")
        return None
    else:
        edges = edge_mask(img, line_size, blur_value)
        return edges

def generate_cartoon(img, total_color):
    if img is None:
        print("Failed to load image.")
        return None
    else:
        img = color_quantization(img, total_color)
        return img

def apply_bilateral_filter(img, d, sigmaColor, sigmaSpace):
    if img is None:
        print("Failed to load image.")
        return None
    else:
        bilateral = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        return bilateral
