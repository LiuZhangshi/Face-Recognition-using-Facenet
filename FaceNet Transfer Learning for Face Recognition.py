#!/usr/bin/env python
# coding: utf-8

# This code borrowed the pre-trained FaceNet model from Hiroki Taniai
# https://github.com/nyoki-mtl/keras-facenet
# The model's structure adopts combinations of Inception module and ResNet

# In[1]:


import numpy as np
from numpy import expand_dims
from numpy import asarray

from PIL import Image
from matplotlib import pyplot

from keras.models import load_model

from os import listdir

# Before face recognition, we need to do face detection in photos and localize the face (assumes 1 face 1 photo)
# for croping and resizing.
# Here we use MTCNN for face detection
from mtcnn.mtcnn import MTCNN


# In[2]:


model = load_model('facenet_keras.h5')


# In[3]:


model.summary()


# In[4]:


# convert cropped face to a 128-dimensional vector
# Hiroki Taniai model requires the input images to be 'rgb', standardized and have a shape of (160, 160)
# Sides: in statistics, standardization is the process of putting different variables on the same scale.
def preprocess_face(face, model):
    face = face.astype('float32')
    
    # standardize pixel values
    mean, std = face.mean(), face.std()
    face_pixels = (face - mean) / std
    
    # The imported face has shape (px, py, 3)
    # The input fed to the model is required to be shape (m, px, py, 3)
    face_pixels = expand_dims(face_pixels, axis = 0)
    vec = model.predict(face_pixels)
    return vec
    
    
# extract a single face from a given photograph
# Hiroki Taniai's model requires the input size to be (160, 160)
def extract_face(filename, required_size=(160, 160)):
    # load PIL image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to numpy array
    pixels = asarray(image)
    
    # detect faces in the image using MTCNN
    # the outputs are bounding boxes
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize img to (160,160)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load faces from a folder containing list of photos
# and use the facenet model to convert faces to 128-dimensional vectors
def load_faces(directory, model):
    # create a dict for (name: vec) mapping
    face_vec = {}
    for filename in listdir(directory):
        if filename.endswith(".jpg"):
            path = directory + filename
            face = extract_face(path)
            vec = preprocess_face(face, model)
            name = filename
            face_vec[name] = vec 
    return face_vec

folder = 'image_people/'
face_vec = load_faces(folder, model)


# In[5]:


# This is an application example
# I create a pool of all avengers's face embedding vectors
# I also downloaded some 'visitors' pictures
# By comparing the visitor's vector with the vectors in the pool, 
# we can judge whether the visitor belongs to avenger or not

def whether_avengers(filename, face_vec, model):
    visitor_face = extract_face(filename, required_size=(160, 160))
    pyplot.imshow(visitor_face)
    visitor_encode = preprocess_face(visitor_face, model)
    min_dist = 100
    for name in face_vec.keys():
        dist = np.linalg.norm(face_vec[name] - visitor_encode)
        if dist < min_dist:
            min_dist = dist
            identity = name.split('.')[0]
    # if the distance is greater than a certain shreshold, then not an avenger
    if min_dist > 11:
        print("Not an Avenger!")
    else:
        print("Welcome back, " + str(identity))
        print("Distance is " + str(min_dist))
    return min_dist, identity


# In[6]:


import numpy as np
image_visitor = 'camera/thor_visitor.jpg'
whether_avengers(image_visitor, face_vec, model)


# In[ ]:




