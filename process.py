import face_recognition
import cv2
import numpy as np

def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process(image1, image2):
    if isinstance(image1, bytes):
        image1 = imread_buffer(image1)
    if isinstance(image2, bytes):
        image2 = imread_buffer(image2)
    face1_encoding = face_recognition.face_encodings(image1)[0]
    face2_encoding = face_recognition.face_encodings(image2)[0]
    face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
    return {
        'distance': face_distance,
        'matched': 'True' if face_distance < 0.6 else 'False',
        'matched_strict': 'True' if face_distance < 0.5 else 'False'
    }
