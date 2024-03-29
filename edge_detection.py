import cv2
import numpy as np

# Read the original image
img = cv2.imread('paper_2.png')
# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
# ----------------------------------------- Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)  # Combined X and Y Sobel Edge Detection
img_sobel = sobelx + sobely + sobelxy
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X+y+XY', img_sobel)
cv2.waitKey(0)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)

# --------------------------------- Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=100)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
# ------------------------------------------

# ------------------------------------------ prewitt
n = 16
kernelx = np.array([[1, n, 1], [0, 0, 0], [-1, -n, -1]])
kernely = np.array([[-1, 0, 1], [-n, 0, n], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_blur, -1, kernelx)
img_prewitty = cv2.filter2D(img_blur, -1, kernely)
cv2.imshow("Prewitt X", img_prewittx)
cv2.waitKey(0)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.waitKey(0)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.waitKey(0)

all = edges + img_prewittx + img_prewitty
cv2.imshow("all", all)
cv2.waitKey(0)

cv2.destroyAllWindows()
