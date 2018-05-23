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


(image, imageW, imageH) = readArguments()





