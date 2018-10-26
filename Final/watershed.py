import numpy as np
import cv2 as cv
import random
from otsu import OtsuFastMultithreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

for name_idx in range(1, 22):
	name = str(name_idx)
	img = cv.imread('Images/new/'+name+'.jpg')
	image = cv.resize(img,(0,0),fx=0.2,fy=0.2)
	reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

	silhouette = []

	for k in range(2,6):
		kmeans = KMeans(n_clusters=k, n_init=40, max_iter=500).fit(reshaped)
		silhouette.append(silhouette_score(reshaped, kmeans.labels_))

	# optimal k
	k = silhouette.index(max(silhouette)) + 1
	print("k = ", k)

	# k = 1

	b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	otsu = OtsuFastMultithreshold()
	otsu.load_image(gray)
	kThresholds = otsu.calculate_k_thresholds(k)

	crushed = otsu.apply_thresholds_to_image(kThresholds)
	# cv.imshow("crushed", crushed)
	# cv.waitKey(0)
	# exit(0)

	otsu = OtsuFastMultithreshold()
	otsu.load_image(b)
	kThresholds = otsu.calculate_k_thresholds(k)
	thresholds = np.array(kThresholds)
	# print thresholds
	otsu = OtsuFastMultithreshold()
	otsu.load_image(g)
	thresholds += otsu.calculate_k_thresholds(k)
	# print thresholds
	otsu = OtsuFastMultithreshold()
	otsu.load_image(r)

	# temp = otsu.calculate_k_thresholds(k)
	thresholds += otsu.calculate_k_thresholds(k)
	# print thresholds

	for i in range(0, thresholds.shape[0]):
		thresholds[i] /= 3

	kThresholds = thresholds 
	print(kThresholds)
	num_segments = len(kThresholds) + 1
	segments = np.zeros(shape = [num_segments, gray.shape[0], gray.shape[1]], dtype = np.uint8)

	# num_segments = int(num_segments/2) + 1
	for k in range(0,num_segments):
		if(k == 0):
			segments[k][gray < kThresholds[0]] = 255
		elif(k == num_segments - 1):
			segments[k][gray >= kThresholds[k - 1]] = 255
		else:
			segments[k][gray >= kThresholds[k - 1]] = 255
			segments[k][gray >= kThresholds[k]] = 0

	unknown = np.zeros(gray.shape)
	sure_fg_gl = np.zeros(gray.shape)

	# K = [0, 1, 2]

	for k in range(0, num_segments-1):
		kernel = np.ones((3, 3), np.uint8)
		# opening = cv.morphologyEx(segments[k],cv.MORPH_OPEN,kernel, iterations = 2)
		opening = segments[k]
		sure_bg = cv.dilate(opening, kernel, iterations = 2)
		dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
		ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)
		# sure_fg = cv.erode(opening, kernel, iterations = 3)
		sure_fg = np.uint8(sure_fg)
		unknown += cv.subtract(sure_bg, sure_fg)
		# cv.imshow('result', sure_fg)
		# cv.waitKey(0)
		# temp = random.randint(1, 100)
		# if (temp%2 == 0):
		sure_fg_gl += sure_fg



	# markers = np.ones(gray.shape)
	# sure_fg_gl = cv.erode(sure_fg_gl, kernel, iterations = 3)
	# cv.imshow('sfg', sure_fg_gl)
	# cv.waitKey(0)
	# cv.erode(unknown, kernel, iterations = 1)

	sure_fg_gl = np.uint8(sure_fg_gl)
	ret, markers = cv.connectedComponents(sure_fg_gl)
	# markers = markers*255/(markers.max() - markers.min())
	# cv.imshow('markers', unknown)
	# cv.waitKey(0)
	markers = markers + 1
	markers[unknown == 255] = 0;

	markers = cv.watershed(img, markers)
	result = np.zeros(gray.shape)
	# result[markers == -1] = 255
	# result = img
	# result[unknown== 255] = (255,0,0)
	result[markers == -1] = 255
	# cv.imshow('result', result)
	# cv.waitKey(0)
	cv.imwrite('Images/outputs/'+ name + '_'+str(k)+'.jpg', result)