import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.image as mpimg 

import imageio
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical

#Each image has 784 pixels so we resize the image to 28x28
def resizeImage(image):
    height = 28
    width = 28
    resizedImage = image.resize((width, height))
    return resizedImage
      
#Image in which the only colors are shades of gray
#makes it easier to translate it to pixels 
def greyscaleImage(image):
    transformedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return transformedImage
    
    
#Adds a blurring filter to reduce image noise    
def denoiseImage(image):
    transformedImage = cv2.GaussianBlur(image, (5, 5), 0)
    return transformedImage
    
#Segementation (thresholding)   
#binarize a grascale image based on pixel intensities using Otsu’s auto technique
#The output is a binary image.
def binarizeImage(image):
   transformedImage = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   return transformedImage

    
def preprocessImage(imagePath):
    image = cv2.imread(imagePath)
    transformedImage1 = resizeImage(image)
    transformedImage2 = greyscaleImage(transformedImage1)
    transformedImage3 = denoiseImage(transformedImage2)
    transformedImage4 = binarizeImage(transformedImage3)
    

#Reading training data beginning at the first row
trainLetters = pd.read_csv(r"C:\Users\mostafalotfy\Desktop\OCRFinalProject\Dataset\TrainData\csvTrainImages 60k x 784.csv")
X_train = trainLetters.values[:,0:]
#Reading training labels-only the first column
trainLabels = pd.read_csv(r"C:\Users\mostafalotfy\Desktop\OCRFinalProject\Dataset\TrainData\csvTrainLabel 60k x 1.csv")
y_train = trainLabels.iloc[:, 0]
y_train = y_train.to_numpy()

testLetters = pd.read_csv(r"C:\Users\mostafalotfy\Desktop\OCRFinalProject\Dataset\TestData\csvTestImages 10k x 784.csv")
X_test = testLetters.values[:,0:]
testLabels = pd.read_csv(r"C:\Users\mostafalotfy\Desktop\OCRFinalProject\Dataset\TestData\csvTestLabel 10k x 1.csv")
y_test = testLabels.iloc[:, 0]
y_test = y_test.to_numpy()


#(X_train[0],cmap=plt.cm.binary)
#Before Normalization ,its a grey image and all the values inside is from 0-255
#print(X_train[0])

#Normalization zero is column and one is the row
X_train =tf.keras.utils.normalize(X_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)
#plt.imshow(X_train[0],cmap=plt.cm.binary)
#After Normalization,it divided all the values by 255
print(X_train[0])

print(y_train[0])#just to check that we have lables inside the network for each corrosponding data in X_train

#After Normalizating we increase one dimeension for the keranl operations,To make it suitable for convolution operation
IMG_SIZE=28
X_trainr=np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_testr=np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print("Training samples Dimension",X_trainr.shape)
print("Testing samples Dimension",X_testr.shape)
#Creating a deep NN

model=Sequential()

#First Convolution layer
model.add(Conv2D(64,(3,3),input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))#to make it non linear ,<0 remove and greater than zero will allow..ro move to the 2nd layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Second convolutional layer
model.add(Conv2D(64,(3,3),input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))#to make it non linear ,<0 remove and greater than zero will allow..ro move to the 2nd layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Third convolutional layer
model.add(Conv2D(64,(3,3),input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))#to make it non linear ,<0 remove and greater than zero will allow..ro move to the 2nd layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Fully connected layer
model.add(Flatten())
model.add(Dense(64))#all neurons are connected 
model.add(Activation("relu"))

model.add(Dense(32))#all neurons are connected 
model.add(Activation("relu"))

model.add(Dense(10))#all neurons are connected 
model.add(Activation("softmax"))



model.summary()
#compiling
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
#training step data- label 
model.fit(X_trainr,y_train,epochs=1,validation_split=0.3)
#if the accuracy is much greater than the validation accuarcy then we will have an overfitting problem that might be fixed by adding dropout layer 

#Evalyating on testing the dataset
test_loss,test_acc=model.evaluate(X_testr,y_test)
print("Test Loss on 10,000 test samples",test_loss)
print("Validation accuarcy on 10,000 test samples",test_acc)

#Prediction part 
predicions=model.predict([X_testr])
print(predicions)
print(np.argmax(predicions[8]))

# index 8 OUTPUT = 9


im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()



gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray.shape
resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
resized.shape

#from keras.models import load_model
#model = load_model("test_model.h5")

#prediction = model.predict(gray)
#print(prediction.argmax())


newimg=tf.keras.utils.normalize(resized,axis=1)
newimg=np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)
newimg.shape

predicion=model.predict(newimg)
print(np.argmax(predicion))
