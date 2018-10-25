import numpy as np
import cv2 as cv
from otsu import OtsuFastMultithreshold

# img = cv.imread('images/1.jpg')
img = cv.imread('/home/sanskar/Academics/IP_Assignment/Term_Project/images/BSDS300/images/train/24004.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
otsu = OtsuFastMultithreshold()
otsu.load_image(gray)
kThresholds = otsu.calculate_k_thresholds(2)
print(kThresholds)
crushed = otsu.apply_thresholds_to_image(kThresholds)
cv.imshow('threshold', crushed)

num_segments = len(kThresholds) + 1
segments = np.zeros(shape = [num_segments, gray.shape[0], gray.shape[1]], dtype = np.uint8)
for k in range(0, num_segments):
	if(k == 0):
		segments[k][gray < kThresholds[0]] = 255
	elif(k == num_segments - 1):
		segments[k][gray >= kThresholds[k - 1]] = 255
	else:
		segments[k][gray >= kThresholds[k - 1]] = 255
		segments[k][gray >= kThresholds[k]] = 0

unknown = np.zeros(gray.shape)
sure_fg_gl = np.zeros(gray.shape)

for k in range(0, num_segments):
	kernel = np.ones((3,3), np.uint8)
	opening = cv.morphologyEx(segments[k],cv.MORPH_OPEN,kernel, iterations = 2)
	# opening = segments[k]
	sure_bg = cv.dilate(opening, kernel, iterations = 1)
	dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
	ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)
	# sure_fg = cv.erode(opening, kernel, iterations = )
	sure_fg = np.uint8(sure_fg)
	unknown += cv.subtract(sure_bg, sure_fg)
	if(k%2 == 0):
		sure_fg_gl += sure_fg
	# cv.imshow('sure_bg', sure_bg)
	# cv.imshow('sure_fg_gl', sure_fg_gl)
	# cv.waitKey(0)



# markers = np.ones(gray.shape)
# sure_fg_gl = cv.erode(sure_fg_gl, kernel, iterations = 3)
# cv.imshow('sfg', sure_fg_gl)
# cv.waitKey(0)
# unknown = cv.erode(unknown, kernel, iterations = 1)
cv.imshow('sure_fg_gl', sure_fg_gl)
cv.imshow('unknown', unknown)
sure_fg_gl = np.uint8(sure_fg_gl)
ret, markers = cv.connectedComponents(sure_fg_gl)
markers = markers + 1
markers[unknown == 255] = 0;

markers = cv.watershed(img, markers)
result = np.zeros(gray.shape)
result[markers == -1] = 255
cv.imshow('result', result)
cv.waitKey(0)
cv.imwrite('images/temp_out.jpg', result)
