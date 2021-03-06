import numpy as np
import cv2
import pandas as pd
from image_descripter import ColorDescriptor
from PIL import Image
import time

class ColorBasedSearcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def get_features_from_query_image(self, image_path):
        queryFeatures = []
        return queryFeatures

    def search(self, query_image_path, limit=3):
        index_file = pd.read_csv(self.indexPath)
        index_file_numpy = index_file.to_numpy()
        index_array = index_file_numpy[:, 4:]
        start = time.time()
        cd = ColorDescriptor((8, 12, 3))
        query_features = np.array(cd.describe(cv2.imread(query_image_path)))
        print("Image dimension:", cv2.imread(query_image_path).shape)
        print("Color Feature Extraction Time = ", time.time()-start)
        # query_feature_array = np.tile(np.array(query_features), (index_array.shape[0], 1))
        distance_dict = {}
        for i in  range(0, index_array.shape[0]):
            distance = np.linalg.norm(index_array[i, :]-query_features)
            distance_dict[i] = distance

        sorted_dict = sorted(distance_dict.items(), key=lambda x: x[1])
        selected = sorted_dict[:limit]
        selected_index = [x[0] for x in selected]
        selected_path = []
        selected_images = []
        for i in selected_index:
            selected_path.append(index_file_numpy[i, 1])
            selected_images.append(cv2.resize(cv2.imread(index_file_numpy[i, 1]), (512, 512)))
        print(selected_index)

        original_img = cv2.resize(cv2.imread(query_image_path), (512, 512))
        upper_joined = np.hstack((original_img, selected_images[0]))
        lower_joined = np.hstack((selected_images[1], selected_images[2]))
        joined_img = np.vstack((upper_joined, lower_joined))

        joined_img = Image.fromarray(cv2.cvtColor(joined_img, cv2.COLOR_BGR2RGB))
        joined_img.show()


        return selected_images

    def nupy_to_csv(self):
        data_file = np.load(r"C:\Users\PC\Downloads\cifar10.npy", allow_pickle=True)
        print(data_file)
        total_features = []
        for i in range(0, data_file.shape[0]):
            name = data_file[i, 512]
            image_path = "D:\GRAD\\2020Spring\MachineLearning_CSC7333\CBIRProject\images\cifar10\\"+name
            features = list(data_file[i, :512])
            features.insert(0, name)
            features.insert(0, name.split("_")[0])
            features.insert(0, image_path)
            total_features.append(features)


        dataframe = pd.DataFrame(total_features)
        dataframe.to_csv(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\CIFAR_image_index_vgg.csv")

c = ColorBasedSearcher(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index.csv")
# c.search(r"C:\Users\PC\Downloads\marguerite-daisy-beautiful-beauty.jpg")
# c.nupy_to_csv()