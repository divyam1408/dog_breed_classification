from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import joblib
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard,CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K



UPLOAD_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def inception_resnet_model(input_shape=(256,256,3),dropout=0):


    '''This function creates our model which we want to train, we are using InceptionResnetV2 model
      as a backbone and using 512 unit Dense layer followed by 120 unit dense layer along with dropouts
      between them'''

    inception_resnet = InceptionResNetV2(include_top=False,input_shape=input_shape)
    inception_resnet.trainable = False

    model = Sequential()
    model.add(inception_resnet)

    model.add(GlobalAveragePooling2D())
    if(dropout!=0):
      model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
    if(dropout!=0):
      model.add(Dropout(0.3))
    model.add(Dense(120,activation='softmax'))

    return model


def infer_model(model, img_path):

  '''This function gives us the prediction for the given data point'''

  img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

  img = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
  img = img/255.




  prediction = model.predict(img[np.newaxis,:,:,:])

  label_enc = joblib.load(os.path.join(os.getcwd(), 'label_enc'))

  return label_enc.classes_[np.argmax(prediction[0])]



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        print('file' not in request.files)
        print('here')
        # if 'file' not in request.files:
        #     #flash('No file part')
        #     return redirect(request.url)
        file = request.files['file1']
        print(file.filename)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if('model' not in os.listdir()):

                model = inception_resnet_model((400,400,3),0.3)

                '''Loading weights for our model make sure that the weights loaded file has same name as is saved in the drive'''
                model_wts = os.path.join(os.getcwd(), 'dog_inception_resnet_4 2')
                model.load_weights(model_wts)
                joblib.dump(model, 'model')

            else:
                model = joblib.load(os.path.join(os.getcwd(), 'model'))


            print('file_name: ',filename)
            img_path = os.path.join(os.getcwd(), 'static',filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            print(img.shape)
            img = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
            img = img/255.
            label_enc = joblib.load('label_enc')
            pred = model.predict(img[np.newaxis,:,:,:])

            print(label_enc.classes_[np.argmax(pred[0])])
            pred = label_enc.classes_[np.argmax(pred[0])]
            return render_template('i.html', user_imag =  os.path.join(app.config['UPLOAD_FOLDER'],filename), prediction = pred,
            actual = filename.split('.')[0])
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))

    return render_template('land1.html')

@app.route('/infer', methods=['GET', 'POST'])
def img_upload():
    if request.method == 'POST':
        return render_template('land1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
   app.run()
