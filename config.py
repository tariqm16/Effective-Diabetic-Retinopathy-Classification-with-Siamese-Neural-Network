# import the necessary packages
import os
# specify the shape of the inputs for our network
IMG_SHAPE = (224, 224, 3)
# specify the batch size and number of epochs
BATCH_SIZE = 8
EPOCHS = 120


# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
#MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "acc_loss_plot.png"])
PLOT_PATH_1 = os.path.sep.join([BASE_OUTPUT, "acc_loss_plot_1.png"])