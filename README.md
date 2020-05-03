# CBIR-ML_Project

This is the source code for the ML-Group-Project for class CS7333.

Evaluating the Feature Extraction Methods with SVM:
    Run ROC.py in terminal with the path of index file as input argument
    Example: python ROC.py "D:\CBIR\image_index.csv"

Retrieval of result image from CBIR:

    Step 1: Download the image collecgtion from the link below:
        https://drive.google.com/open?id=1bvOMZIG5zDUgdegtkuecVvqhfCsBsblf
    Step 2: Build image index of the downloaded image using Index.py
    
    Step 3: After indexing is completed, run SearchUsingVGGandColor.py with the query image
        Example: SearchUsingVGGandColor.py "D:\CBIR\images\query_image.png"


Libraries Requirements:
numpy
pandas
matplotlib
torch == 1.4.0
torchvision == 0.5.0
sklearn == 0.22.1
 