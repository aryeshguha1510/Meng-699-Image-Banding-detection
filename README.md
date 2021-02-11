## DBI (Deep Banding Index)

This is the official **Python Tensorflow** implementations of our ICASSP 2021 paper [*"CAPTURING BANDING IN IMAGES: DATABASE CONSTRUCTION AND
OBJECTIVE ASSESSMENT"*](https://ece.uwaterloo.ca/~z70wang/publications/icassp21_banding.pdf).


- [1. Brief Introduction](#1-brief-introduction)
  * [1.1 Backgrounds](#11-backgrounds)
  * [1.2 Contributions](#12-contributions)
  * [1.3 Results](#13-results)
  * [1.4 Citation](#14-citation)
- [2. Dataset](#2-dataset)
  * [2.1 Banding Patches Dataset](#2-dataset)
  * [2.2 HD Images Dataset with Banded and NonBanded Region Information](#2-dataset)
- [3. Prerequisite](#3-prerequisite)
  * [3.1 Environment](#31-environment)
  * [3.2 Packages](#32-packages)
  * [3.3 Pretrained Models](#33-pretrained-models)
- [4. Relevant Source Codes](#4-relevant-source-codes)
- [5. Codes for comparing models](#5-codes-for-comparing-models)
- [6. Demo](#6-demo)


### 1. Brief Introduction
 The [`Deep Banding Index Paper`](https://ece.uwaterloo.ca/~z70wang/publications/icassp21_banding.pdf) introduces first of its kind dataset[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4513740.svg)](https://doi.org/10.5281/zenodo.4513740), [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4512571.svg)](https://doi.org/10.5281/zenodo.4512571) and an Objective Assesment of Banding quantification in HD images with Deep Banding Index, which aims to capture the annoyance caused to end users (QoE) as they percieve media featuring Banding Artifacts.

#### 1.1 Backgrounds
Banding, colour banding, or false contours is a common visual artifact appearing in images and videos, often in large regions of low textures and slow gradients such as sky. When the granularity of bit-depth or display intensity levels mismatches with the visual system’s perception of the smooth transition of color and luminance presented in the image content,the discontinuity positions in smooth image gradients are transformed into perceivable, wide, discrete bands. Banding significantly deteriorates the perceptual quality-of-experience(QoE) of end users. A visual example is shown in the image below where banding artifacts are clearly visible in the sky.
<br>

 ![](/Results_and_visualizations/Banding_Illustration.PNG)

#### 1.2 Contributions
This work is completed as a research project at University of Waterloo, Under guidance of Dr.Zhou Wang and Jatin Sapra MEng.

#### 1.3 Results
The following results are obtained using the method described in [`Deep Banding Index Paper`](https://ece.uwaterloo.ca/~z70wang/publications/icassp21_banding.pdf)
 ![](/Results_and_visualizations/Performance_Comparison.PNG)

#### 1.4 Citation
Please cite our paper if you find our model or the [Banding Patches Dataset](https://doi.org/10.5281/zenodo.4513740), [HD Images Dataset with Banded and NonBanded region Information](https://doi.org/10.5281/zenodo.4513740) dataset useful.

A. Kapoor, J. Sapra and Z. Wang, ["Capturing banding in images: database construction and objective assessment,"](https://ece.uwaterloo.ca/~z70wang/publications/icassp21_banding.pdf) IEEE International Conference on Acoustics, Speech and Signal Processing, Jun. 2021.


### 2. Dataset
- The Dataset is generated by extracting frames from more than 600 pristine high-definition (HD) videos, resulting in approximately 1,250+ images of 1920 x 1080 resolution. Banding distortion is be introduced by bit-depth reduction (or dynamic range tone mapping) in luma and chroma channels of image/ video (where bit-depth reduction can also leads to other perceptually visible artifacts). 
- Six levels of quantization are used to enhance the diversity of the dataset, The generation of quantized image frames is explained in [`Data Generation From Pristine Videos`](Dataset-Generation/)

#### 2.1 [HD Images Dataset with Banded and NonBanded region Information](https://zenodo.org/)
  - **Download**: The dataset is available on Zenodo under a Creative Commons Attribution license: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4513740.svg)](https://doi.org/10.5281/zenodo.4513740).
  - This dataset features 1250+ HD images with their quantized version along with XML files which give information about these regions.
  - refer python script: [`Patches Generation from HD images`](src/Generating_patches_from_HD_images.py) for understanding semi automatic labelling procedure, Also the image below illustrates the patches generation from HD images.<br> 
![](src/Patches_Generation.png)


#### 2.2 [Banding Patches Dataset](https://zenodo.org/record/3926181#.Xv4vg3X0kUd)

   - **Download**: The dataset is available on Zenodo under a Creative Commons Attribution license: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4512571.svg)](https://doi.org/10.5281/zenodo.4512571)
   - The Dataset features 169,501 patches of size 235x235. These patches were used to train a CNN classier which can be further used to obtain Deep Banding Index for a Banded Image. The Dataset folder features training, validation, and testing dataset, each such folder further contains banded and nonbanded patches.
   - The distribution of patches in the dataset is explained as below: <br>
![](Results_and_visualizations/DatasetDistribution.PNG)



### 3. Prerequisites

#### 3.1 Environment

The code has been tested on `Ubuntu 18.04` and on `Windows 10` with `Python 3.8` and `tensorflow 2.1`

#### 3.2 Packages

`tensorflow-gpu=2.1`, `statistics`, `pandas`, `numpy`, `python 3.8` and `OpenCV`

#### 3.3 Pretrained Models

  - The Pretrained CNN Classifier model could be found in folder [`pretrained_model`](!pretrained_model/)

### 4. Relevant Source Codes

The [`Source`](src/) folder contains the source files used for generating semi-automatic labelled dataset, CNN_classifier training files and prediction file which can be used for generating Deep Banding index for any image. 
- **XML to CSV:** the mask generated for HD images of size 1920x1080 were stored in XML files, these XML files contain information about the rectangular masking coordinates. These rectangular masking coordinates were generated using [`LabelImg`](https://github.com/tzutalin/labelImg).
    - Refer [`XML_to_CSV`](src/xml_to_csv.py) for the code used to convert XML files region infomation to CSV files.
- **Patches Generation:** The patches are generated using heuristic rules from HD images and the obatined csv file about banded, non banded region information.
   - Refer [`Patche Generation From HD Images`](src/Generating_patches_from_HD_images.py)
- **CNN Classifier Training**: The CNN Classifier is trained using the [`Bannding Patches Dataset`](https://zenodo.org/badge/DOI/10.5281/zenodo.4512571.svg).
  - Refer [`Training Script`](src/train.py) for CNN_classifier Training for Banded vs NonBanded classification tasks.
- **Calculating Scores Using Deep Banding Index**: Deep Banding Index is calculated using CNN_model Classifer and the methodology described in the [`DBI paper`](https://ece.uwaterloo.ca/~z70wang/publications/icassp21_banding.pdf). 
  - Refer [`Deep Banding Index Prediction Script`](src/predict.py) for using DBI for out of sample image.
 - **Generating Banding Visualizations**: The script [`Deep Banding Map Generation`](src/Deep_Banding_Map.py) explains the working of Deep Banding Index by generating Deep Banding Maps for HD images.

### 5. Codes for comparing models

refer [`MATLAB Scripts`](MATLAB-scripts-for-comparison/)

### 6. Demo
Use this folder structure to calculate Deep Banding Index for your images, follow these steps to get Deep Banding Score for HD banded Images:
- Download [`Demo`](Meng-699-Image-Banding-detection/Demo) folder.
- Put the HD images in [`Image Path Folder`](Meng-699-Image-Banding-detection/Demo/Given_image_path/)
- Run [`DBI Predict File`](Meng-699-Image-Banding-detection/Demo/predict.py), make sure you have the following dependencies on your device 
  - [`Tensorflow 2.1, numpy, pandas, Open-CV`], and you have [`CNN_Banded Patch classifer`](Meng-699-Image-Banding-detection/Demo/CNN_classifier/) in the same path as presented in Demo folder.
- open the [`CSV Result File`](Meng-699-Image-Banding-detection/Demo/banding_score_results.csv) to see the results associated with the HD images present in the [`Image Path Folder`](Meng-699-Image-Banding-detection/Demo/Given_image_path/). 

