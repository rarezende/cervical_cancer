# -------------------------------------------------------------------------------------- #
# Cervical Cancer Screening
# -------------------------------------------------------------------------------------- #

import time
import numpy as np
import skimage.io as io
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

startTime = time.time()


rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/Data"

sourcePath = rootDir + "/128x128/train/Type_1/*.jpg"
coll = io.imread_collection(sourcePath)
X_1 = io.concatenate_images(coll)
y_1 = np.ones(X_1.shape[0],dtype=int)

sourcePath = rootDir + "/128x128/train/Type_2/*.jpg"
coll = io.imread_collection(sourcePath)
X_2 = io.concatenate_images(coll)
y_2 = 2*np.ones(X_2.shape[0],dtype=int)

sourcePath = rootDir + "/128x128/train/Type_3/*.jpg"
coll = io.imread_collection(sourcePath)
X_3 = io.concatenate_images(coll)
y_3 = 3*np.ones(X_3.shape[0],dtype=int)

X_train = np.concatenate([X_1, X_2, X_3])
y_train = np.concatenate([y_1, y_2, y_3])
#y_train = np_utils.to_categorical(y_train)


def create_model():
    cnn = Sequential()
    cnn.add(Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=X_train.shape[1:]))
    cnn.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    '''
    cnn.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    '''
    cnn.add(Flatten())
    cnn.add(Dense(64, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, activation="sigmoid"))

    cnn.compile(loss="mean_squared_logarithmic_error", optimizer="adam", metrics=['categorical_accuracy'])
    
    return cnn

classifier = KerasClassifier(build_fn = create_model, epochs=2, batch_size=32, verbose=True)

scores = cross_val_score(classifier, X_train, y_train, cv=4, scoring = "accuracy")

print("Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))

print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))




