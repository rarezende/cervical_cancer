# -------------------------------------------------------------------------------------- #
# Convolutional Neural Network Training
# -------------------------------------------------------------------------------------- #

import time
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import rmsprop, Nadam, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------- #
# Definition of Convolutional Neural Network

def create_convnet(inShape):
    cnn = Sequential()
    cnn.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=inShape))
    cnn.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Flatten())
    cnn.add(Dense(1024, activation="relu"))
    cnn.add(Dropout(0.5))
    #cnn.add(Dense(256, activation="relu"))
    #cnn.add(Dropout(0.5))
    cnn.add(Dense(3, activation="softmax"))
    
    opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    #opt = Adam()
    
    cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    
    return cnn

 #%%
# -------------------------------------------------------------------------------------- #
# Creation of training and test datasets

startTime = time.time()

#rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_work/128x128"
rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_all/128x128"

sourcePath = rootDir + "/train/Type_1/*.jpg"
imgColl = io.imread_collection(sourcePath)
X_1 = io.concatenate_images(imgColl)
y_1 = 0*np.ones(X_1.shape[0], dtype=int)

sourcePath = rootDir + "/train/Type_2/*.jpg"
imgColl = io.imread_collection(sourcePath)
X_2 = io.concatenate_images(imgColl)
y_2 = 1*np.ones(X_2.shape[0], dtype=int)

sourcePath = rootDir + "/train/Type_3/*.jpg"
imgColl = io.imread_collection(sourcePath)
X_3 = io.concatenate_images(imgColl)
y_3 = 2*np.ones(X_3.shape[0], dtype=int)

X_all = np.concatenate([X_1, X_2, X_3])
X_all = X_all.astype('float32')
X_all /= 255
y_all = np.concatenate([y_1, y_2, y_3])
y_all = np_utils.to_categorical(y_all)


# -------------------------------------------------------------------------------------- #
# Training with data augmentation

datagen = ImageDataGenerator(
    featurewise_center=False,               # set input mean to 0 over the dataset
    samplewise_center=False,                # set each sample mean to 0
    featurewise_std_normalization=False,    # divide inputs by std of the dataset
    samplewise_std_normalization=False,     # divide each input by its std
    zca_whitening=False,                    # apply ZCA whitening
    rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=0,
    zoom_range=0.1,     
    width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,                   # randomly flip images
    vertical_flip=False)                     # randomly flip images


def scheduler(epoch):
    if epoch == 10:
        K.set_value(model.optimizer.lr, 3e-5)
    elif epoch == 20:
        K.set_value(model.optimizer.lr, 1e-5)
    elif epoch == 40:
        K.set_value(model.optimizer.lr, 3e-6)
    elif epoch == 60:
        K.set_value(model.optimizer.lr, 1e-6)
        
    return K.get_value(model.optimizer.lr)

lr_scheduler = LearningRateScheduler(scheduler)

test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=31416)

model = create_convnet(X_train.shape[1:])

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

batch_size = 32
epochs = 60

# Fit the model on the batches generated by datagen.flow().
steps_per_epoch=X_train.shape[0] // batch_size
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=(X_test, y_test),
                              workers=8,
                              callbacks=[lr_scheduler])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.minorticks_on()
plt.ylim((0.68,1.04))
plt.yticks(np.arange(0.68,1.04,0.02))
plt.grid(b=True, which='major', color='black', linestyle='--')
plt.show()


print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))


#%%
# -------------------------------------------------------------------------------------- #
# Create Submission File

import pandas as pd

sourcePath = rootDir + "/test/*.jpg"
nameTrailer = rootDir + '/test\\'

imgColl = io.imread_collection(sourcePath)
X_test = io.concatenate_images(imgColl)
X_test = X_test.astype('float32')
X_test /= 255

y_test = model.predict_proba(X_test)

subFiles =[]
fileNames = imgColl.files
for fileName in fileNames:
    subFiles.append(str(fileName).replace(nameTrailer, ''))

dfFileNames = pd.DataFrame({'image_name': subFiles})
dfProbs = pd.DataFrame({'Type_1': y_test[:,0], 'Type_2': y_test[:,1], 'Type_3': y_test[:,2]})
submission = pd.concat((dfFileNames, dfProbs), 1)

submission.to_csv("C:/Users/rarez/Documents/Data Science/cervical_cancer/submission.csv", index = False)

   

#%%
# -------------------------------------------------------------------------------------- #
# Training without data augmentation

startTime = time.time()

batch_size = 32
epochs = 50
lr = 0.1

classifier = create_convnet(X_train.shape[1:], lr)

classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))



