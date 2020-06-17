from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from Eyefilter import eyeFilter
import os

# Menu
menuOptions = "[1] Image Classification " \
              "\n[2] Face Filter For Human Emotions" \
              "\n[3] Exit" \
              "\nPlease choose 1/2/3"

# Image Classification function
model = load_model('goodModel55.h5')
IMG_SIZE = 100

def convert_to_array(img):
    im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    return np.array(image)

def get_object_name(label):
    if label==0:
        return "Person"
    if label==1:
        return "Motorbike"
    if label==2:
        return "Fruit"
    if label==3:
        return "Dog"
    if label==4:
        return "Cat"
    if label==5:
        return "Car"

def predict_object(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    a=[]
    a.append(ar)
    a=np.array(a).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    object=get_object_name(label_index)
    print(object)
    print("The predicted Object is a "+object+" with accuracy =    "+str(acc))

def ui():
    objects = input("Please input the file to be predict: ")

    if os.path.exists(objects):
        print(objects)
        predict_object(objects)
    else:
        print("File does not exist!")

    runMenu()

# Face filter based on emotion detection function
# The file classifier depends on the location of the file in the directory
face_classifier = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
classifier = load_model('emotionModel.h5')

emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

def faceFilter():
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class
                # don't forget to input the correct directory of the filter
                preds = classifier.predict(roi)[0]
                label = emotion_labels[preds.argmax()]
                if label == 'Happy':
                    happyFilter = eyeFilter("Filter/sparkle.png", frame)
                    frame = happyFilter.getfilter()
                elif label == 'Angry':
                    angryFilter = eyeFilter("Filter/fire.jpg", frame)
                    frame = angryFilter.getfilter()
                elif label == 'Sad':
                    sadFilter = eyeFilter("Filter/tear.png", frame)
                    frame = sadFilter.getfilter()
                elif label == 'Surprise':
                    surpriseFilter = eyeFilter("Filter/x.png", frame)
                    frame = surpriseFilter.getfilter()
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('Emotion Detector', frame)

        # Press key 'q' to exit from the camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Emotion Detector', 4) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def runFaceFilter():
    faceFilter()
    runMenu()


def runMenu():
    print(menuOptions)
    choice = input("Enter your choice: ")
    if choice=="1":
         ui()
    elif choice=="2":
        runFaceFilter()
    elif choice=="3":
        print("Goodbye!")
    else:
        print("Please choose from the options!")
        runMenu()


runMenu()
