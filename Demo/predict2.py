# -*- coding: utf-8 -*-
"""
@author: Akshay Kapoor
"""

# Importing relevant Libraries
import os
import cv2
import numpy as np
import tensorflow as tf

# Image_data class for converting image into numpy array after scaling the pixel values
class Image_data:
    
    def __init__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        self.img = img        
        
    def np_data(self):
        return self.img


# This class is used for extracting neighborhood patches of size 128x128 from 256x256 images
class Shape_work:
    
    # dictionaries to store patch information
    dict_series1 = {}

    # gathering size of image
    def __init__(self, h, w):
        self.h = h
        self.w = w
        
    # iterating over the 256x256 image, extracting 128x128 patches
    def iterate_patch(self):
        """
        Uses h : height of image, 
             w: width of image
        Returns:
            A list containing the starting coordinates of 128x128 patches
            and initializes the patch information in a dictionary.
        """
        patch_size = 256
        h_step = int(self.h / patch_size)
        w_step = int(self.w / patch_size)

        series1 = []
        dict_series1 = self.dict_series1
        
        for i in range(h_step):
            for j in range(w_step):
                element = [i * patch_size, j * patch_size]
                identifier = f"{element[0]}_{element[1]}"
                series1.append(element)
                dict_series1[identifier] = [1, 1000]
                
        return [dict_series1, series1]
    

def h_w_neighbours_index(h, w, patch_size=256):
    """
    Calculates the coordinates of the 8 neighboring patches surrounding a given patch.
    """
    ListO = [h, h + patch_size, w, w + patch_size]  # Original patch
    ListR = [h, h + patch_size, w + patch_size, w + 2 * patch_size]  # Right neighbor
    ListL = [h, h + patch_size, w - patch_size, w]  # Left neighbor
    ListT = [h - patch_size, h, w, w + patch_size]  # Top neighbor
    ListB = [h + patch_size, h + 2 * patch_size, w, w + patch_size]  # Bottom neighbor
    ListTL = [h - patch_size, h, w - patch_size, w]  # Top-left neighbor
    ListTR = [h - patch_size, h, w + patch_size, w + 2 * patch_size]  # Top-right neighbor
    ListBL = [h + patch_size, h + 2 * patch_size, w - patch_size, w]  # Bottom-left neighbor
    ListBR = [h + patch_size, h + 2 * patch_size, w + patch_size, w + 2 * patch_size]  # Bottom-right neighbor
    
    return ListTL, ListT, ListTR, ListL, ListO, ListR, ListBL, ListB, ListBR

# Main function
if __name__ == "__main__":
    
    # Path to images folder
    file_folder = "./DBIrepo/Meng-699-Image-Banding-detection/Demo/Given_image_path/"
    
    # Initializing dictionary to store scores for each image
    dict_score = {}

    # Set alpha, beta, gamma (weights for the main patch and neighbors)
    alpha = 32 / 40
    beta = 1 / 40
    gamma = 1 / 40

    # Image dimensions for 256x256 input images
    h_shape = 256
    w_shape = 256
    
    # Getting patch information for the image size
    image_shape_object = Shape_work(h_shape, w_shape)
    image_series = image_shape_object.iterate_patch()

    iter1 = image_series[0], image_series[1] 

    # Loading the trained CNN patches model
    model = tf.keras.models.load_model('.\DBIrepo\Meng-699-Image-Banding-detection\Demo\CNN_classifier')

    # Iterate over the files in the folder
    for file in os.listdir(file_folder):
        
        # Read the file data and extract pixel data as numpy array
        path = os.path.join(file_folder, file)
        print(path)
        obj = Image_data(path)
        image_data = obj.np_data()

        # Iterate over the series of patches
        for iter_object in iter1[1]:
            for item in iter_object:
                # Get the patch coordinates
                h = item[0]
                w = item[1]

                # Get the coordinates of the neighbors
                element9_list = h_w_neighbours_index(h, w)

                # Initialize a list to store CNN predictions for each neighbor
                element9_preds = []

                # Iterate over each neighbor
                for element in element9_list:
                    h1_temp = element[0]
                    w1_temp = element[2]

                    # Check if the neighbor is already visited
                    if iter1[0][f"{h1_temp}_{w1_temp}"][0] == 1:
                        
                        # Update the neighbor information
                        patch = image_data[h1_temp:h1_temp + 128, w1_temp:w1_temp + 128]
                        patch = np.expand_dims(patch, axis=0)
                        
                        # Get model prediction for the given patch
                        result = model.predict(patch)[0]
                        
                        # Thresholding result (banded = 0, nonbanded = 1)
                        ans = 1 - result
                        ans = 1 if ans > 0.2 else 0

                        # Append CNN model predictions for all neighbors
                        element9_preds.append(ans)

                    else:
                        ans = iter1[0][f"{h1_temp}_{w1_temp}"][1]
                        element9_preds.append(ans)

                # Accumulating answers using alpha, beta, and gamma
                final_ans = alpha * element9_preds[4] 
                final_ans += beta * (element9_preds[1] + element9_preds[3] + element9_preds[5] + element9_preds[7])
                final_ans += gamma * (element9_preds[0] + element9_preds[2] + element9_preds[6] + element9_preds[8])

                # Update the score of the visited patch
                iter1[0][f"{h}_{w}"] = [0, final_ans]

            total_sum = sum([iter1[0][i][1] for i in iter1[0]])
            overall_sum = total_sum  # total sum for 256x256 image
