import cv2
import numpy as np

def read_image(path):
    return cv2.imread(path)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def detect_edges(image, threshold1=100, threshold2=200):
    return cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)
