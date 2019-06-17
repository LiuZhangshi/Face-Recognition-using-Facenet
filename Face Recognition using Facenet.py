#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model


# In[2]:

# load pretrained facenet model.
# This model transfer the image to a 128 vector, namely face embedding.
model = load_model('facenet_keras.h5')


# In[3]:


model.summary()




# In[5]:
# define triplet loss for one shot learning

def triplet_loss(y, yhat, alpha = 0.2):
    anchor, positive, negative = yhat[0], yhat[1], yhat[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


# In[6]:


model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

# load pre-trained weights
model.load_weights('facenet_keras_weights.h5')


# In[56]:


from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# preprocess image and face embedding
def preprocess_face(face, model):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face_pixels = (face - mean) / std
    face_pixels = expand_dims(face_pixels, axis = 0)
    vec = model.predict(face_pixels)
    return vec
    
    
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	#using MTCNN to detect face in the image
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load faces to form a directory of items you would like to recognize.
def load_faces(directory, model):
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

print(face_vec)

# In[57]:

# Recognize the face from the camera. Check whether it's the one in your directory.
def whether_avengers(filename, face_vec, model):
    visitor_face = extract_face(filename, required_size=(160, 160))
    pyplot.imshow(visitor_face)
    visitor_encode = preprocess_face(visitor_face, model)
    min_dist = 100
    for name in face_vec.keys():
        dist = np.linalg.norm(face_vec[name] - visitor_encode)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 10:
        print("Not an Avenger!")
    else:
        print("Welcome back, " + str(identity))
        print("Distance is " + str(min_dist))
    return min_dist, identity



# In[70]:


import numpy as np

# Stark asked us to build a face recognition system for the avengers.
# If it's an avenger, the system will let her/him in, otherwise let them go.
# Here we go! Avengers!

image_visitor = 'camera/ironman.jpg'
whether_avengers(image_visitor, face_vec, model)
