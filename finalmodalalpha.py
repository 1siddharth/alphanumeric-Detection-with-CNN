from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout,BatchNormalization
import numpy as np 
import sys
import os
import tensorflow as tf
import cv2
from random import shuffle

import numpy as np
#tf.logging.set_verbosity(tf.logging.info)

IMG_SIZE=40
TRAIN_DIR = r"C:\Users\Siddharth\Desktop\uav stuff\New folder\New folder\generate_targets-master\train"
TEST_DIR = r"C:\Users\Siddharth\Desktop\uav stuff\New folder\New folder\generate_targets-master\test"

def revalimg(img):
   y=0
   x=0      
   crop_img = img[y:y-5, x:x-5]
   crop_img=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
   crop_img = cv2.resize(crop_img, (IMG_SIZE,IMG_SIZE))
   crop_img= np.reshape(crop_img,1*IMG_SIZE*IMG_SIZE)
   return crop_img
   #cv2.imshow("cropped", crop_img)

h=[0]*36
def one_hot(c):
  i=h[:]
  if(ord(c)>=65):
    i[ord(c)-65]=1
  else:
    i[ord(c)+26-48]=1
  return i


def Label_train():
  trainthis1=[]
  path=os.listdir(TRAIN_DIR)
  for x in path:
     ap=os.path.join(TRAIN_DIR,x)
     path2=os.listdir(ap)
     for xx in path2:    
        ap2=os.path.join(ap,xx)
        img2=cv2.imread(ap2)
        ft=revalimg(img2)
        abc=ap2.split("\\")[-2]
        hothot=one_hot(abc)

        trainthis1.append([np.array(ft),np.array(hothot)])
                
       
        shuffle(trainthis1)
        print("done1")
        #print(abc)
  np.save("trainednet.npy",trainthis1)
  return trainthis1



def Label_test():
  testthis1=[]
  path=os.listdir(TEST_DIR)
  for x in path:
     ap=os.path.join(TEST_DIR,x)
     path2=os.listdir(ap)
     for xx in path2:    
        ap2=os.path.join(ap,xx)
        img2=cv2.imread(ap2)
        ft=revalimg(img2)
        abc=ap2.split("\\")[-2]
        hothot=one_hot(abc)
        testthis1.append([np.array(ft),np.array(hothot)])
    
        print("done1")
  np.save("testednet.npy",testthis1)
  return testthis1 





IMG_SIZE = 100
trainthis2=np.load("trainednet.npy",allow_pickle=True)
print(trainthis2.shape)

#import matplotlib.pyplot as plt
#img = plt.imshow(trainthis2[349][0])

model = Sequential()

model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(100, 100,1), activation='relu', name='input'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(BatchNormalization())


model.add(Dense(1024, activation='relu'))
model.add(Flatten())
model.add(Dense(35, activation='softmax',name='targets')) 

#os.chdir(r"C:\Users\Siddharth\Desktop\suas final\modal stuff")
trainthis2=np.load("trainednet.npy",allow_pickle=True)
testthis2=np.load("testednet.npy",allow_pickle=True)


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
X1 = np.array([i[0] for i in trainthis2]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y1=[i[1] for i in trainthis2]

X2 =  np.array([i[0] for i in testthis2]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y2 = [i[1] for i in testthis2]

hist = model.fit([X1],[Y1],batch_size=11, epochs=10,verbose=1,validation_data=([X2],[Y2]))

os.chdir(r"C:\Users\Siddharth\Desktop\suas final")
model.save("ok.keras")
           
