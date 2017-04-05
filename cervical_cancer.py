from skimage.io import imread, imshow
from skimage.transform import resize
import skimage.io as io
import numpy as np


root_dir = "C:/Users/rarez/Documents/Data Science/Cervical_Cancer"
train1_dir = "/Train/Type_1/"

image_file = root_dir + train1_dir + "13*.jpg" 

coll = io.ImageCollection(image_file, conserve_memory = True)

img_resized = np.zeros(shape=(len(coll),640,640,3))
i=0

for img in coll:
    img_resized[i] = resize(img, (640,640), mode='constant')
    i = i+1
    
    

img = imread(image_file)
io.imshow(img)

img_resized = resize(img, (640,640), mode='reflect')

io.imshow(coll[0])
io.imshow(img_resized[0])


coll[0].shape

coll[10].shape

trainset = coll.concatenate()


