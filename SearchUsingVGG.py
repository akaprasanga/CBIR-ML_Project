import numpy as np
import pandas as pd
from vgg_net import VGGNetFeat
from PIL import Image
import time
import cv2
from SearchUsingColor import ColorBasedSearcher

class SearchUsingVGG():

    def __init__(self, index_file_path, search_image_path):
        self.index_file_path = index_file_path
        self.search_image_path = search_image_path
        pass

    def get_features_of_query_image(self, search_image_path):
        s = time.time()
        print("VGG Feature Extration Started...")
        vgg_obj = VGGNetFeat()
        features = vgg_obj.run_inference(search_image_path)
        print("VGG Feature Extraction Completed. Time taken=",time.time()-s)
        return features

    def search(self, search_image_path, limit=3):
        index_file = pd.read_csv(self.index_file_path)
        index_file_numpy = index_file.to_numpy()
        index_array = index_file_numpy[:, 4:]
        query_features = self.get_features_of_query_image(search_image_path)
        # query_feature_array = np.tile(np.array(query_features), (index_array.shape[0], 1))
        distance_dict = {}
        for i in range(0, index_array.shape[0]):
            distance = np.linalg.norm(index_array[i, :] - query_features)
            distance_dict[i] = distance

        sorted_dict = sorted(distance_dict.items(), key=lambda x: x[1])
        selected = sorted_dict[:limit]
        selected_index = [x[0] for x in selected]
        selected_path = []
        selected_images = []
        for i in selected_index:
            selected_path.append(index_file_numpy[i, 1])
            selected_images.append(cv2.resize(cv2.imread(index_file_numpy[i, 1]), (512, 512)))

        original_img = cv2.resize(cv2.imread(search_image_path), (512, 512))
        upper_joined = np.hstack((original_img, selected_images[0]))
        lower_joined = np.hstack((selected_images[1], selected_images[2]))
        joined_img = np.vstack((upper_joined, lower_joined))

        joined_img = Image.fromarray(cv2.cvtColor(joined_img, cv2.COLOR_BGR2RGB))
        joined_img.show()


        return selected_images

image_path_to_search = r"C:\Users\PC\Desktop\38843.jpg"

colorObj = ColorBasedSearcher(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index.csv")
colorObj.search(image_path_to_search)

vggObj = SearchUsingVGG(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index_vgg.csv", image_path_to_search)
vggObj.search(image_path_to_search)
