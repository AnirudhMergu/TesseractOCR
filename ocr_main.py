import os
import numpy as np
from PIL import Image,ImageTk
import PIL
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense,Flatten
from keras.models import Sequential, load_model
from cropyble import Cropyble
import cv2
import shutil
import time

location = input("Enter the location of the image : ")
image = cv2.imread(location,0)
thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

result = cv2.GaussianBlur(thresh, (5,5), 0)
result = 255 - result

cv2.imwrite('final_img.png',result)
loc = 'final_img.png'


if os.path.isfile('DATA/train/.DS_Store'):
    os.remove('DATA/train/.DS_Store')
if os.path.isfile('DATA/test/.DS_Store'):
    os.remove('DATA/test/.DS_Store')
    
for i in range(1,80):
    if os.path.isfile(f'DATA/train/{i}/.DS_Store'):
        os.remove(f'DATA/train/{i}/.DS_Store')
    if os.path.isfile(f'DATA/test/{i}/.DS_Store'):
        os.remove(f'DATA/test/{i}/.DS_Store')

datagen = ImageDataGenerator(rescale = 1./255,
                            zoom_range = 0.2)


trained_image = datagen.flow_from_directory('DATA/train',
                                            target_size = (32,32),
                                            class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_image = test_datagen.flow_from_directory('DATA/test',
                                            target_size = (32,32),
                                            class_mode = 'categorical')


def Train_Model():
    global trained_image, test_image
    model = Sequential()
    conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (32,32,3))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(26,activation = 'softmax'))
    model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(trained_image, epochs = 30, validation_data = test_image, validation_steps = 1)
    model.save('model.h5')
    
Train_Model()


def Predict_Model(img):
    global trained_image
    new_model = load_model('model.h5')
    img = cv2.resize(img,(32,32),3)
    img = np.expand_dims(img,axis=0)
    img = img / 255
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0',',',';',':','?','!','.','@','#','$','%','&','(',')','{','}','[',']']
    
    prediction = new_model.predict_classes(img)
    prediction = prediction[0]
    
    my_dict = dict(trained_image.class_indices)
    
    for key,value in my_dict.items():
        if prediction == value:
            return key


def Word_Extract(location):
    
    if os.path.isdir('WORDS'):
        shutil.rmtree('WORDS')
        os.mkdir('WORDS')
    else:
        os.mkdir('WORDS')
        
    my_img = Cropyble(location)
    img = PIL.Image.open(location)
    
    words = my_img.get_words()
    select = []
    for i in range(len(words)):
        if (words[i] != '' or words[i] != ' '):
            if (len(words[i]) > 1):
                select.append(words[i])
            elif (len(words[i]) == 1):
                if ( i != 0):
                    if (len(words[i-1]) > 1 and len(words[i+1]) > 1):
                        select.append(words[i])
                elif ( i == 0):
                    select.append(words[i])

                    
    j = 0

    for i in select:
        rect = my_img.get_box(i)
        crop_img = img.crop((rect[0]-20,rect[1]-20,rect[2]+20,rect[3]+20))
        crop_img.save(f'WORDS/{j}.png')
        j += 1
    
    return my_img, select
      
    
cropyble_img,select = Word_Extract(loc)


if os.path.isdir('LETTERS'):
    shutil.rmtree('LETTERS')
    os.mkdir('LETTERS')
else:
    os.mkdir('LETTERS')
    
    
for i in range(len(os.listdir('WORDS'))):
    os.mkdir(f'LETTERS/{i}')
    image = cv2.imread(f"WORDS/{i}.png") 
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(image, 10, 250) 
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    idx = 0 
    for c in cnts: 
        x,y,w,h = cv2.boundingRect(c) 
        if w>50 and h>50: 
            idx+=1 
            new_img=image[y:y+h,x:x+w] 
            cv2.imwrite(f"LETTERS/{i}/" + str(idx) + '.png', new_img) 
            


text = ''
for i in range(len(os.listdir('LETTERS'))):
    string = ''
    char_dict = {}
    for j in range(len(os.listdir(f'LETTERS/{i}'))):
        img = cv2.imread(f'LETTERS/{i}/{j+1}.png')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        char = Predict_Model(img)
        
        
        for l in range(len(select[i])):
            if char == select[i][l]:
                char_dict[select[i].index(char)] = char
                
                
            else:
                for k in range(len(select[i])):
                    if k != l:
                        if char == select[i][k]:
                            char_dict[k] = char 
        
        keys = list(char_dict.keys())
        if char not in select[i]:
            for val in range(len(select[i])):
                if val not in keys:
                    char_dict[val] = char

    keys = list(char_dict.keys())
    keys.sort()
    #print(keys)
    
    for m in keys:
        string += char_dict[m]
        
        
    string += ' '
    text += string
              

for i in range(10):
    print("")
print("The predicted sentence is: ",text)
    