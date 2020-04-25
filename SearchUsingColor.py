import numpy as np
import cv2
import pandas as pd
from image_descripter import ColorDescriptor
from PIL import Image
class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def get_features_from_query_image(self, image_path):
        queryFeatures = []
        return queryFeatures

    def search(self, query_image_path, limit=3):
        index_file = pd.read_csv(self.indexPath)
        index_file_numpy = index_file.to_numpy()
        index_array = index_file_numpy[:, 4:]
        cd = ColorDescriptor((8, 12, 3))
        query_features = np.array(cd.describe(cv2.imread(query_image_path)))
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
            selected_images.append(cv2.imread(index_file_numpy[i, 1]))
        print(selected_index)

        Image.open(query_image_path).show(title="original")
        # cv2.imshow("original",cv2.imread(query_image_path))
        # cv2.waitKey(0)

        for i,images in enumerate(selected_path):
            Image.open(images).show(title=str(i))
            # cv2.imshow(str(i), images)
            # cv2.waitKey(0)

    # def chi2_distance(self, histA, histB, eps= 1e-10):
    #     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
    #                       for (a, b) in zip(histA, histB)])
    #
    #     return d

    def nupy_to_csv(self):
        data_file = np.load(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\CBIR_VGG_index.npy", allow_pickle=True)
        print(data_file)
        total_features = []
        for i in range(0, data_file.shape[0]):
            name = data_file[i, 512]
            image_path = "D:\\GRAD\\2020Spring\MachineLearning_CSC7333\CBIRProject\images\collection\\"+name
            features = list(data_file[i, :512])
            features.insert(0, name)
            features.insert(0, name.split("_")[0])
            features.insert(0, image_path)
            total_features.append(features)


        dataframe = pd.DataFrame(total_features)
        dataframe.to_csv(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index_vgg.csv")

c = Searcher(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index.csv")
c.search(r"C:\Users\PC\Downloads\marguerite-daisy-beautiful-beauty.jpg")
# c.nupy_to_csv()