import numpy as np
import pandas as pd
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\cifar-10-batches-py\data_batch_1")
# print(images)

cifar_labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
keys = list(data.keys())
labels = data[keys[1]]
images_array = data[keys[2]]
for i, img in enumerate(images_array):
    reshaped = np.array(img).reshape((3, 32,32))
    array = reshaped.transpose(1,2,0)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    cv2.imwrite("D:\GRAD\\2020Spring\MachineLearning_CSC7333\CBIRProject\images\cifar10\\"+cifar_labels[labels[i]]+"_"+str(i)+".png", array)
    print(i)

print(images_array)