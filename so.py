
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology

# img = cv2.imread('so4.jpg',0)
img = cv2.imread('test2.tif',0)

# original thresholding
# ret1,th1 = cv2.threshold(img,15,255,cv2.THRESH_BINARY)
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# mask out circle
# height, width = img.shape
# circle_img = np.zeros((height,width), np.uint8)
# cv2.circle(circle_img,(height//2,width//2),200,50,thickness=-1)
# img = cv2.bitwise_and(img, img, mask=circle_img)

# take care of edges
# img[img < 74] = 0

# histogram calculation
raveled = img.ravel()

# thresholding
img = cv2.GaussianBlur(img,(3,3),0)
threshold, dom = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print(threshold)
# dom = cv2.adaptiveThreshold(dom, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 20)

# mask out again
# dom = cv2.bitwise_and(dom, dom, mask=circle_img)

# erode - works bad
# kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
# dom = cv2.erode(dom,kernel,iterations = 1)

# dilate 3x3 kernel
# kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
# dom = cv2.dilate(dom, kernel, iterations=1)

# dilate 5x5 kernel
# k = [
#     [0,0,1,0,0],
#     [0,1,1,1,0],
#     [1,1,1,1,1],
#     [0,1,1,1,0],
#     [0,0,1,0,0]
# ]
# kernel = np.array(k, np.uint8) 
# dom = cv2.dilate(dom, kernel, iterations=1)

# plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.hist(raveled[raveled > 74],256)
# plt.subplot(222),plt.hist(raveled,256)

plt.subplot(223),plt.imshow(img,'gray')
plt.subplot(224),plt.imshow(dom,'gray')

plt.show()
