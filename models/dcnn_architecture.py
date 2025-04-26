import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense

num_classes = 8

def build_model():
    
    model = Sequential(name='DCNN')
    
    # Input dimensions from the global variables
    img_width, img_height, img_depth = IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS
    num_classes = NUM_CLASSES
    
    # First convolutional block
    model.add(Conv2D(64, (5,5), activation='elu', padding='same', 
                    input_shape=(img_width, img_height, img_depth), 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), activation='elu', padding='same', 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    # Second convolutional block
    model.add(Conv2D(128, (3,3), activation='elu', padding='same', 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='elu', padding='same', 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    # Third convolutional block
    model.add(Conv2D(256, (3,3), activation='elu', padding='same', 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='elu', padding='same', 
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model