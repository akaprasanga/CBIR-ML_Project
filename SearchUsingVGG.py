import numpy as np
import pandas as pd
from vgg_net import VGGNetFeat
from PIL import Image
import time

class SearchUsingVGG():

    def __init__(self, index_file_path, search_image_path):
        self.index_file_path = index_file_path
        self.search_image_path = search_image_path
        pass

    def get_features_of_query_image(self):
        s = time.time()
        print("VGG Feature Extration Started...")
        vgg_obj = VGGNetFeat()
        features = vgg_obj.run_inference(self.search_image_path)
        print("VGG Feature Extraction Completed. Time taken=",time.time()-s)
        return features

    def search(self, limit):
        index_file = pd.read_csv(self.index_file_path)
        index_file_numpy = index_file.to_numpy()
        index_array = index_file_numpy[:, 4:]
        query_features = self.get_features_of_query_image()
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
            # selected_images.append(cv2.imread(index_file_numpy[i, 1]))
        print(selected_index)

        Image.open(self.search_image_path).show(title="original")
        # cv2.imshow("original",cv2.imread(query_image_path))
        # cv2.waitKey(0)

        for i,images in enumerate(selected_path):
            Image.open(images).show(title=str(i))
            # cv2.imshow(str(i), images)
            # cv2.waitKey(0)

c = SearchUsingVGG(r"D:\GRAD\2020Spring\MachineLearning_CSC7333\CBIRProject\image_index_vgg.csv", r"D:\GRAD\2020Spring\Semiconductor_EE7260\FinalDefects\images\20190226-bridgetisrael08.jpg")
c.search(5)
