# Effective-Diabetic-Retinopathy-Classification-with-Siamese-Neural-Network
Python Code for Effective Diabetic Retinopathy Classification with Siamese Neural Network: A Strategy for Small Dataset Challenges

## Compatibility
The code is tested using Python 3.8.18 and Tensorflow version 2.13.1.

## Requirements:
The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

<code>```pip install -r requirements.txt```</code>

## Datasets:
Two Datasets have been used in this paper: the FGADR dataset and Aptos 2019 Blindness Detection.
The FGADR dataset has fine-grained annotations on 1842 fundus images with pixel- and image-level labels. The dataset is taken from the
UAE hospitals and is the property of the Inception Institute of Artificial Intelligence, Abu Dhabi. I
The sample images from the dataset with lesion details are given below:
![Alt Text](https://github.com/tariqm16/Effective-Diabetic-Retinopathy-Classification-with-Siamese-Neural-Network/blob/main/Images/Siamese_model_Classes.png)
few_shot_labels.csv file includes the images used in the training of the FGADR dataset.

The Aptos 2019 dataset can be downloaded from here: (https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
The sample images from the dataset are given below:
![Alt Text](https://github.com/tariqm16/Effective-Diabetic-Retinopathy-Classification-with-Siamese-Neural-Network/blob/main/Images/Siamese_model_Aptos_classes.png)

## Code Execution:
The code is executed by:
<code>```python train_siamese_network.py```</code>
The training architecture of the model can be seen below:
![Alt Text](https://github.com/tariqm16/Effective-Diabetic-Retinopathy-Classification-with-Siamese-Neural-Network/blob/main/Images/Siamese_model_Architecture.png)

## Model Evaluation:
The model performance is assessed using different performance metrics:
### Accuracy: 
Accuracy gives the percentage of correctly predicted labels out of the total number of samples. To evaluate the accuracy of our model, the predicted labels are compared
with the ground truth labels for a given set of images. 

### Quadratic Weighted Kappa: 
QuadraticWeighted Kappa (QWK) takes into account the agreement between predicted and actual labels, adjusted for the possibility of chance agreement. It considers class imbalance by applying class weights.
