# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skimage import metrics
import cv2
from tensorflow.keras.applications.resnet import preprocess_input
from siamese_network_update import build_siamese_model
import config


def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def cal_ssim(imageA,imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score = metrics.structural_similarity(grayA, grayB, data_range=grayB.max() - grayB.min())
    return score

def make_all_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    idx=[]
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))

    for i in range(0, numClasses):
        idx.append(find_indices(labels, i))

    # loop over all images
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append(1)
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        # randomly select four negative classes
        negClasses = np.random.choice(np.delete(np.unique(labels), label), 4, replace=False)

        negImages = []
        neg_labels=[]
        for negClass in negClasses:
            negIndices = np.where(labels == negClass)[0]
            temp_images=np.take(images, negIndices, axis=0)
            neg_random=np.random.choice(negIndices)
            neg_labels.append(neg_random)
            negImage = images[neg_random]
            negImages.append(negImage)

        negDist=[]
        for negImage in negImages:
            # find distances between negative image and anchor image
            negDist.append(np.linalg.norm(negImage-currentImage))
        hard_negative = neg_labels[np.argmax(negDist)]

        # add hardest negative pair to the pairs list
        pairImages.append([currentImage, images[hard_negative]])
        pairLabels.append(0)
    return (np.array(pairImages), np.array(pairLabels))


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    idx=[]
    # calculate the total number of classes present in the dataset and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))

    for i in range(0, numClasses):
        idx.append(find_indices(labels, i))
    # loop over all images
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    return (np.array(pairImages), np.array(pairLabels))
	
def vector_distance(imageA,imageB):
    imageA=np.expand_dims(imageA, axis=0)
    imageB=np.expand_dims(imageB, axis=0)
    featureExtractor = build_siamese_model((config.IMG_SHAPE))
    featsA = featureExtractor(imageA)
    featsB = featureExtractor(imageB)
    distance=euclidean_distance([featsA,featsB])
    return distance
                     
    
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))
	

def cosine_distance(vectors):
    # Compute the cosine similarity between the two vectors
    (featsA, featsB) = vectors
    dot = K.sum(featsA * featsB, axis=1,keepdims=True)
    mag_x = K.sqrt(K.sum(K.square(featsA), axis=1,keepdims=True))
    mag_y = K.sqrt(K.sum(K.square(featsB), axis=1,keepdims=True))
    cos_sim = dot / (mag_x * mag_y + K.epsilon())
    return 1 - cos_sim
    
def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)