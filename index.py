from image_descripter import ColorDescriptor
import cv2
import glob
import pandas as pd
import os
from PIL import Image

def describe(path):

	cd = ColorDescriptor((8, 12, 3))

	output = open("color_features.csv", 'w')
	total_features = []
	for imageName in os.listdir(path):
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imagePath = path+"\\"+imageName
		# imageID = imagePath.split('\\')[-1]
		image = cv2.imread(imagePath)
		# image = cv2.Canny(image, 50, 100)
		# cv2.imwrite('edges.jpg', image)
		# image = cv2.imread('edges.jpg')
		features = cd.describe(image)
		features.insert(0, imageName)
		features.insert(0, imageName.split("_")[0])
		features.insert(0, imagePath)
		total_features.append(features)
		# features = [str(f) for f in features]
		# output.write("%s,%s\n" % (imageID, ",".join(features)))


	dataframe = pd.DataFrame(total_features)
	dataframe.to_csv(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index.csv")

def rename(path):
	i = 0
	for imagePath in os.listdir(path):
		# image = cv2.imread(path+"\\"+imagePath[i])
		try:
			image = Image.open(path+"\\"+imagePath)
			image.save(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\images\collection\\"+path.split("\\")[-1]+"_"+str(i)+".png")
			# cv2.imwrite(str(i)+'edges.jpg', image)
			i += 1
		except:
			continue
	print("finished processing")

# rename(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\images\nature")
describe(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\images\collection\\")