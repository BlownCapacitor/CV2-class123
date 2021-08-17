import numpy as np 
from sklearn.datasets import fetch_openml
import time
from cv2 import cv2  
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import seaborn as sns
import ssl

print('image.npz')
print('labels.csv')
x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(x)
print(y)
if (not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
letterclasses = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
number_of_classes = len(letterclasses)
samples_per_class = 5

figure = plt.figure(figsize = (number_of_classes*2,(1+ samples_per_class*2)))
idx_class = 0

for cls in letterclasses:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_class, replace = False)
  i = 0
  for idx in idxs:
    plt_idx = i*number_of_classes +idx_class + 1
    p = plt.subplot(samples_per_class, number_of_classes, plt_idx)
    p = sns.heatmap(np.reshape(x[idx],(22,30)),cmap = plt.cm.gray, xticklabels= False, yticklabels = False, cbar = False) 
    p = plt.axis('off')
    i += 1
  idx_class += 1

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 7500, test_size = 2500, random_state = 9)

x_trainScale = x_train/255.0
x_testScale = x_test/255.0

LR = LogisticRegression(solver = 'saga', multi_class= 'multinomial').fit(x_trainScale, y_train)

lrPrediction = LR.predict(x_testScale)

accuracy = accuracy_score(lrPrediction, y_test)

print(accuracy)

confusionMatrix = pd.crosstab(y_test, lrPrediction, rownames= ['actual'], colnames= ['predicted'])
p = plt.figure(figsize= (10,10))
p = sns.heatmap(confusionMatrix, annot= True, fmt = 'd', cbar = False)

capture = cv2.VideoCapture(0)

while True:
    try: 
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape 
        upper_left = (int(width / 2 - 56), int(height / 2 - 56)) 
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56)) 
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L') 
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized) 
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784) 
        test_pred = LR.predict(test_sample) 
        print("Predicted class is: ", test_pred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break  
    except Exception as e: pass

capture.release() 
cv2.destroyAllWindows()
