# IS-Final-Project

# By Chan Elizabeth W, Gadiza Namira Putri Andjani, and Winson Wardana

- The main program code is presented in the main.py

- The code for training the model is presented in the 2 Jupyter notebook. (trainimgModel.ipynb and trainemotionModel.ipynb)

- The haarcascade_frontalface_default.xml is used to detect the frontal face. It is a stump-based 24x24 discrete adaboost frontal face detector. It is created by Rainer Lienhart.

- The shape_predictor_68_face_landmarks.dat is a pre-trained facial landmark detector inside the dlib library used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face.

      The indexes of the 68 coordinates can be visualized on the image: https://www.pyimagesearch.com/wp-     content/uploads/2017/04/facial_landmarks_68markup-1024x825.jpg

- The Eyefilter.py is the code for coordinating and storing the eye positions from the dlib facial landmark detector. It is used to locate the eyes of the person for filter.

- The Additional Pre trained model file contain our additional trained models. One for the emotion detection and the other one for the image classification.

- The Filter file contain the images that we used as filter for the face filter based on emotion detection.

- The ImageTest file contatin the images that we take from google and we used to test the model prediction based on the image classification.
