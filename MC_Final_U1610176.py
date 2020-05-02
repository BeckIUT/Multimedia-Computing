import cv2
import numpy as np

img = cv2.imread('images/lane-line-road-9.jpg', cv2.IMREAD_GRAYSCALE) # reading an image in Grayscale

h,w = img.shape # getting the height and width of an image

high = 255 # assigning high
low = 150  # assigning low before contrast stretching

contrasted = np.array(img) # using new variable to work with image

for i in range(0,h):	# contrast stretching
	for k in range(0,w):
		if contrasted[i][k]<=low:
			contrasted[i][k] = 0
		elif contrasted[i][k]>low and contrasted[i][k]<=high:
			contrasted[i][k] = 255*(contrasted[i][k]-low)/(high-low)
		elif contrasted[i][k]>high:
			contrasted[i][k]=255

kernel = np.ones((3,3), np.uint8)
builtInClosing = cv2.morphologyEx(contrasted, cv2.MORPH_CLOSE, kernel) # built-in opencv function to do closing operation on an image

originalCanny = cv2.Canny(img, 150, 255) # Canny on original image
contrastedCanny = cv2.Canny(contrasted, 150, 255) # Canny on constrasted image
builtInClosingCanny = cv2.Canny(builtInClosing, 150, 255) # Canny after Closing operation on contrasted image

laplacian1 = cv2.Laplacian(img, cv2.CV_64F)
laplacian2 = cv2.Laplacian(contrasted, cv2.CV_64F)
laplacian3 = cv2.Laplacian(builtInClosing, cv2.CV_64F)

cv2.imshow('original', img) # showing original image
cv2.imshow('contrast-stretched', contrasted) # showing contrast-stretched image
cv2.imshow('built-in-closing-after-contrast', builtInClosing) # showing closing image of contrast-stretched image

cv2.imshow('canny-on-original', originalCanny) # showing Canny detection on original image
cv2.imshow('canny-after-builtInClosing', builtInClosingCanny) # showing Canny Detection after Closing operation on contrasted image
cv2.imshow('canny-on-contrast', contrastedCanny) # showing Canny detection on constrasted image

cv2.imshow('laplace-original', laplacian1) # showing original image
cv2.imshow('laplace-contrast-stretched', laplacian2) # showing contrast-stretched image
cv2.imshow('laplace-built-in-closing-after-contrast', laplacian3) # showing closing image of contrast-stretched image
cv2.waitKey(0)