import cv2
import numpy as np
import dlib
from math import hypot

# Make the eye coordinates using Dlib face detector for the filter in the emotion detection
class eyeFilter:
    def __init__(self, image, frame):
        self.image = image
        self.frame = frame

    def getfilter(self):
        rows, cols, _ = self.frame.shape
        e_mask = np.zeros((rows, cols), np.uint8)

        eye_image = cv2.imread(self.image)

        # Loading Face detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        e_mask.fill(0)

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:

            landmarks = predictor(gray, face)

            # First Eye coordinates
            top_eye = ((landmarks.part(19).x + 7), (landmarks.part(19).y + 10))
            center_eye = ((landmarks.part(37).x + 14), (landmarks.part(37).y - 5))
            left_eye = ((landmarks.part(36).x - 7), (landmarks.part(36).y + 10))
            right_eye = ((landmarks.part(39).x - 10), (landmarks.part(39).y + 10))

            # Second Eye coordinates
            top_eye2 = ((landmarks.part(23).x + 7), (landmarks.part(23).y + 10))
            center_eye2 = ((landmarks.part(43).x + 14), (landmarks.part(43).y - 5))
            left_eye2 = ((landmarks.part(42).x + 7), (landmarks.part(42).y + 10))
            right_eye2 = ((landmarks.part(45).x + 10), (landmarks.part(45).y + 10))

            eye_width = int(hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]) * 1.7)
            eye_height = int(eye_width * 0.67)

            eye_width2 = int(hypot(left_eye2[0] - right_eye2[0], left_eye2[1] - right_eye2[1]) * 1.7)
            eye_height2 = int(eye_width2 * 0.67)

            # New nose position
            top_left = (int(center_eye[0] - eye_width / 2), int(center_eye[1] - eye_height / 2))
            bottom_right = (int(center_eye[0] + eye_width / 2), int(center_eye[1] + eye_height / 2))

            top_left2 = (int(center_eye2[0] - eye_width2 / 2), int(center_eye2[1] - eye_height2 / 2))
            bottom_right2 = (int(center_eye2[0] + eye_width2 / 2), int(center_eye2[1] + eye_height2 / 2))


            # Adding the new eye
            eye_pair = cv2.resize(eye_image, (eye_width, eye_height))

            eye_pair2 = cv2.resize(eye_image, (eye_width2, eye_height2))

            eye_pair_gray = cv2.cvtColor(eye_pair, cv2.COLOR_BGR2GRAY)

            eye_pair_gray2 = cv2.cvtColor(eye_pair2, cv2.COLOR_BGR2GRAY)

            _, eye_mask = cv2.threshold(eye_pair_gray, 25, 255, cv2.THRESH_BINARY_INV)

            _, eye_mask2 = cv2.threshold(eye_pair_gray2, 25, 255, cv2.THRESH_BINARY_INV)

            eye_area = self.frame[top_left[1]: top_left[1] + eye_height, top_left[0]: top_left[0] + eye_width]

            eye_area2 = self.frame[top_left2[1]: top_left2[1] + eye_height2, top_left2[0]: top_left2[0] + eye_width2]

            eye_area_no_eye = cv2.bitwise_and(eye_area, eye_area, mask=eye_mask)

            eye_area_no_eye2 = cv2.bitwise_and(eye_area2, eye_area2, mask=eye_mask2)

            final_eye = cv2.add(eye_area_no_eye, eye_pair)

            final_eye2 = cv2.add(eye_area_no_eye2, eye_pair2)

            self.frame[top_left[1]: top_left[1] + eye_height, top_left[0]: top_left[0] + eye_width] = final_eye

            self.frame[top_left2[1]: top_left2[1] + eye_height2, top_left2[0]: top_left2[0] + eye_width2] = final_eye2

        return self.frame
