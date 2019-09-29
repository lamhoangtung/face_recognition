import base64
import io
import json

import cv2
import numpy as np
import requests
from PIL import Image

import face_recognition


def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_image_base64(base64string):
    image = Image.open(io.BytesIO(base64.b64decode(base64string)))
    image.save('./test.jpg')
    return image


def get_profile_image_from_layout(id_img):
    layout_api_address = 'http://35.240.219.152:8989/profile_image'
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    _, img_encoded = cv2.imencode('.jpg', id_img)
    response = requests.post(layout_api_address, data=img_encoded.tostring(),
                             headers=headers)
    response = json.loads(response.text)
    list_face = []
    for each in response['prediction']:
        list_face.append(read_image_base64(each['cropped']))
    return list_face


def error(message):
    return {
        'success': False,
        'message': message
    }


def process(id_img, selfie_img):
    if isinstance(id_img, bytes):
        id_img = imread_buffer(id_img)
    if isinstance(selfie_img, bytes):
        selfie_img = imread_buffer(selfie_img)

    # Call drake's API to get the cropped profile_image
    list_id_face = get_profile_image_from_layout(id_img)
    if not list_id_face:
        return error("Can't find any face in the ID. Please take a new picture of your ID")
    elif len(list_id_face) > 1:
        return error("Multiple faces have been found. Please take a new picture of only your ID")

    # Check number of face in selfie images
    list_selfie_face = face_recognition.api.face_locations(selfie_img)
    if not list_selfie_face:
        return error("Can't find any face in your selfie. Please take a new picture of you")
    elif len(list_selfie_face) > 1:
        return error("Multiple faces have been found. Please take a new picture of only you")

    # Check hash if they are the same picture
    if list_id_face[0] == list_selfie_face[0]:
        return error('Found the same images. Please take new picture of you and your id')

    # Compare distance
    face1_encoding = face_recognition.face_encodings(list_id_face[0])[0]
    face2_encoding = face_recognition.face_encodings(list_selfie_face[0])[0]
    face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
    return {
        'distance': face_distance,
        'matched': 'True' if face_distance < 0.6 else 'False',
        'matched_strict': 'True' if face_distance < 0.5 else 'False'
    }


if __name__ == "__main__":
    img = cv2.imread('./test_img/1.png')
    print(get_profile_image_from_layout(img))
