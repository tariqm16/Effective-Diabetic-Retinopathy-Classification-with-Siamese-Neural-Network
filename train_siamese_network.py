# import the necessary packages
from siamese_network import build_siamese_model
import config
import utils
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda,Layer
import numpy as np
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import binary_crossentropy
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample
from focal_loss import SparseCategoricalFocalLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    square_margin = K.square(K.maximum(margin - y_pred, 0))
    y_true = K.cast(y_true, 'float32')
    return K.mean(y_true * square_pred + (1 - y_true) * square_margin)

def get_image_patch(image):
    height, width, _ = image.shape
    # Define the size of the crop area
    crop_height = int(height * 0.5)
    crop_width = int(width * 0.5)
    # Get the top-left corner of the crop area
    start_x = int((width - crop_width) / 2)
    start_y = int((height - crop_height) / 2)
    # Get the center crop of the image
    center_crop = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return center_crop

def evaluate_model(model,testX,testY,trainX,trainY):
    y_true = testY  # true labels of test set
    y_pred = []     # predicted labels of test set

    # loop over test set and predict label for each image
    for i in range(len(testX)):
        testX_distance=[]
        for j in range(len(trainX)):
            distance = model.predict([np.expand_dims(testX[i], axis=0), np.expand_dims(trainX[j],axis=0)])
            testX_distance.append(distance)
        nearest_index = np.argmin(testX_distance)
        nearest_label = trainY[nearest_index]
        y_pred.append(nearest_label)

    # compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    QWK_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic') 
    print("QWK_kappa:", QWK_kappa)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    return accuracy,QWK_kappa

def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
    for d in list(data):
        if d=="Image name":
            for i in (data[d].tolist()):
                image= cv2.imread(img_folder+'/'+i, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (224, 224))
                image=preprocess_input(image)
                img_data_array.append(image)
        elif d=="Retinopathy grade":
            for i in (data[d].tolist()):
              class_name.append(i)
    return img_data_array,class_name

def create_aptos_dataset(img_folder):
    img_data_array=[]
    img_name_array=[]
    class_name=[]
    for d in list(data):
        print(d)
        if d=="id_code":
            for i in (data[d].tolist()):
                img_name_array.append(i)
        elif d=="diagnosis":
            for i in (data[d].tolist()):
                class_name.append(i)
    for img in range(len(img_name_array)):
        print(img_folder+'/Class_'+str(class_name[img])+'/'+img_name_array[img]+'.png')
        image= cv2.imread(img_folder+'/Class_'+str(class_name[img])+'/'+str(img_name_array[img])+'.png')#, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (512, 512))
        image=preprocess_input(image)
        img_data_array.append(image)
        
    return img_data_array,class_name


def shuffle_data(pairImages,pairLabels):
    num_pairs = len(pairImages)

    # create an array of indices for the pairs
    pair_indices = np.arange(num_pairs)

    # shuffle the indices randomly
    np.random.shuffle(pair_indices)

    # use the shuffled indices to shuffle the pairs
    shuffled_pairImages = pairImages[pair_indices]
    shuffled_pairLabels = pairLabels[pair_indices]
    return shuffled_pairImages,shuffled_pairLabels


#------------------------------------------------------------------------------------------------------
# Call the BinaryFocalLoss function
binary_focal_loss =tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
# load the dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading FGADR dataset...")
import pandas as pd 
data = pd.read_csv("/home/tariqm16/Siamese Network/few_shot_labels.csv").loc[:,["Image name", "Retinopathy grade"]]

#data = pd.read_csv("/home/tariqm16/Aptos_data/train_aptos.csv").loc[:,["id_code", "diagnosis"]]


#------------------------------------------------------------------------------------------------------------
#img_data, class_name = create_aptos_dataset(r'/home/tariqm16/Aptos_data/')
img_data, class_name = create_dataset(r'/home/tariqm16/Original_Images/Fundus Images/')

train_acc_scores = []
val_acc_scores = []
test_acc_scores=[]
kappa_score=[]
roc_score=[]
n_splits=3


#------------------------------------------------------------------------------------------------------
# first, split into train and test sets, with balanced class distribution
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

train_test_indices = []
for train_index, test_index in skf.split(img_data, class_name):
    train_test_indices.append((train_index, test_index))

# now, perform k-fold cross-validation on the training set, with balanced class distribution
train_val_images = []
train_val_labels = []
#for train_index, val_index in kf.split(train_test_indices[0][0]):
for i in range(n_splits):
    skf_train = StratifiedKFold(n_splits=5, shuffle=True)

    train_images=np.take(img_data, train_test_indices[i][0], axis=0)
    train_labels=np.take(class_name, train_test_indices[i][0], axis=0)
    for train_index, val_index in skf_train.split(np.take(img_data, train_test_indices[i][0], axis=0),np.take(class_name, train_test_indices[i][0], axis=0)):
        train_val_images.append((np.take(train_images, train_index, axis=0), np.take(train_images, val_index, axis=0)))
        train_val_labels.append((np.take(train_labels, train_index, axis=0), np.take(train_labels, val_index, axis=0)))

# Loop through each fold of the cross-validation
for i in range(n_splits):
    #train_index = train_test_indices[i][0]
    test_index = train_test_indices[i][1]
    # Get the data for this fold
    trainX=train_val_images[i][0]
    trainY=train_val_labels[i][0]
    valX=train_val_images[i][1]
    valY=train_val_labels[i][1]
    #trainX = np.take(img_data, train_index, axis=0)
    #trainY = np.take(class_name, train_index, axis=0)    
    testX = np.take(img_data, test_index, axis=0)
    testY = np.take(class_name, test_index, axis=0)    
    # prepare the positive and negative pairs
    print("[INFO] preparing positive and negative pairs...")
    (pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
    (pairVal, labelVal) = utils.make_pairs(valX, valY)
    (pairtest, labeltest) = utils.make_pairs(testX, testY)
    #pairTrain,labelTrain=shuffle_data(pairTrain, labelTrain)

    imgA = Input(shape=config.IMG_SHAPE)
    imgB = Input(shape=config.IMG_SHAPE)
    
    # configure the siamese network
    print("[INFO] building siamese network...")
    #featureExtractor=build_model_with_attention((config.IMG_SHAPE))
    featureExtractor = build_siamese_model((config.IMG_SHAPE))
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    
    # finally, construct the siamese network
    distance = Lambda(utils.cosine_distance)([featsA, featsB])
    output = Dense(1, activation='sigmoid')(distance)

    # Create model
    model = Model([imgA,imgB], output)
    model.summary()
    
    # compile the model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=0.0001)
    #opt = SGD(lr=0.0001)
    model.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])
    checkpoint_filepath = '/home/tariqm16/Siamese Network/output/siamese_model/saved_model.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    lr_plateau=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, mode='max', min_lr=1e-10,verbose=1)
    early = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
    callbacks_list = [checkpoint,early]#,lr_plateau]#,tf.keras.callbacks.CSVLogger('/home/tariqm16/history.csv')]
    
    
    # train the model
    print(tf.config.list_physical_devices('GPU'))
    print("[INFO] training model...")
    print(np.shape(pairTrain[:, 0]))
    history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain, validation_data=([pairVal[:, 0], pairVal[:, 1]], labelVal), batch_size=config.BATCH_SIZE, epochs=80, callbacks=callbacks_list)

    # Evaluate the model on training and validation data
    train_loss, train_acc = model.evaluate([pairTrain[:, 0], pairTrain[:, 1]], labelTrain)
    val_loss, val_acc = model.evaluate([pairVal[:, 0], pairVal[:, 1]], labelVal)
    print("Test data Evaluation......................")
    test_accuracy,QWK_kappa=evaluate_model(model, testX, testY, trainX, trainY)
    print("Test Accuracy:",test_accuracy)
    print("Training accuracy:",train_acc)
    print("Validation accuracy:", val_acc)

    model.save_weights('/home/tariqm16/Siamese Network/output/siamese_model/model_weights.h5')

    train_acc_scores.append(train_acc)
    val_acc_scores.append(val_acc)
    test_acc_scores.append(test_accuracy)
    kappa_score.append(QWK_kappa)
# Compute average training and validation accuracy scores
avg_train_acc = sum(train_acc_scores) / len(train_acc_scores)
avg_val_acc = sum(val_acc_scores) / len(val_acc_scores)
avg_test_acc = sum(test_acc_scores) / len(test_acc_scores)
print("Complete list of test scores:",test_acc_scores)
print("Average training accuracy:", avg_train_acc)
print("Average validation accuracy:", avg_val_acc)
print("Average Test accuracy:", avg_test_acc)
print("Average kappa score:", kappa_score)
s
print("[INFO] saving siamese model...")
model.save(checkpoint_filepath)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH_1)
