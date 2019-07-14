# One-Shot-Face-Recognition-using-Facenet

In this project, we use pretrained FaceNet model (from Hiroki Taniai) to build an application of one-shot face recognition.

The model implements face embedding (convert face image to a 128-dimensional vector) by using a combination structure of Inception module  and ResNet.

The input preprocess requires that the input is an "RGB" face image of size (160, 160). Thus we need to localize the face in the photo for cropping and resizing. Here we use MTCNN for face detection.

An application example is provided: I create a pool of all avengers's face embedding vectors. I also downloaded some 'visitors' pictures. By comparing the visitor's vector with the vectors in the pool, we can judge whether the visitor belongs to avenger or not. This is a one-shot-learning application: we only have 1 picture for each avenger and 1 picture for each visitor. 

Please see references below for the download links of FaceNet and MTCNN.

References:

1. FaceNet: A Unified Embedding for Face Recognition and Clustering, 2015.
2. https://github.com/nyoki-mtl/keras-facenet
3. https://github.com/ipazc/mtcnn
4. https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
