
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test1.tif',0)

ret1,th1 = cv2.threshold(img,15,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_,dom = cv2.threshold(img,70,255,cv2.THRESH_BINARY_INV)

# mask out circle
# height, width = dom.shape
# circle_img = np.zeros((height,width), np.uint8)
# cv2.circle(circle_img,(99,99),88,50,thickness=-1)
# dom = cv2.bitwise_and(dom, dom, mask=circle_img)

# 3x3 kernel
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8) 

# 5x5 kernel
# k = [
#     [0,0,1,0,0],
#     [0,1,1,1,0],
#     [1,1,1,1,1],
#     [0,1,1,1,0],
#     [0,0,1,0,0]
# ]
# kernel = np.array(k, np.uint8) 

# dilate
dom = cv2.dilate(dom, kernel, iterations=1)

plt.subplot(221),plt.imshow(img,'gray')
# plt.subplot(222),plt.imshow(circle_img,'gray')
plt.subplot(223),plt.imshow(blur,'gray')
plt.subplot(224),plt.imshow(dom,'gray')

plt.show()
