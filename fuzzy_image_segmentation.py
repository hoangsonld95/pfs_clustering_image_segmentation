import numpy as np
import sys 
from PIL import Image
from sklearn import preprocessing 
from sklearn.metrics.pairwise import euclidean_distances

iterations = 5

# Read arguments 

def readArguments(): 
	
	if(len(sys.argv) < 4):
		print("Error: Insufficient arguments, take 3 arguments")
		sys.exit()
	else:
		K = int(sys.argv[1])
		if K < 3:
			print("K has to be greater than 2")
			sys.exit()
		inputName = sys.argv[2]
		outputName = sys.argv[3]

	image = Image.open(inputName)
	imageW = image.size[0]
	imageH = image.size[1]

	return (image, imageW, imageH)

def initImageMatrix():

	imageVector = np.ndarray(shape=(imageW * imageH, 5), dtype=float)
	pixelBelongsTo = np.ndarray(shape=(imageW * imageH), dtype=int)

	for y in range(0, imageH):
		for x in range(0, imageW):
			xy = (x, y)
			rgb = image.getpixel(xy)
			imageVector[y * imageW + x, 0] = rgb[0]
			imageVector[y * imageW + x, 1] = rgb[1]
			imageVector[y * imageW + x, 2] = rgb[2]
			imageVector[y * imageW + x, 3] = x
			imageVector[y * imageW + x, 4] = y

	imageVector_scaled = preprocessing.normalize(imageVector)
	
	return 

# Read arguments
(image, imageW, imageH) = readArguments()

# Read the image matrix. normalize each feature of image
initImageMatrix()







