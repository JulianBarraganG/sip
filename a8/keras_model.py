from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam
from const import DATA_FOLDER

## Configure the network

# batch_size to train
batch_size = 20 * 256
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 300

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential([
    InputLayer(shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation="relu"),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    # Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation="relu"),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    # Dropout(0.25),
    Flatten(),
    Dense(units=4000, activation="relu"),
    Dense(units=nb_classes, activation="softmax"),
])
    
optimizer = Adam()

model.compile(optimizer=optimizer,
             loss="categorical_crossentropy",
             metrics=["accuracy"])

## Load the pretrained network
model.load_weights(DATA_FOLDER / "keras_model.h5") 
