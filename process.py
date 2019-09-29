import base64
import io
import json

import cv2
import imagehash
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


def is_same_image(image1, image2):
    hash1 = imagehash.whash(Image.fromarray(image1))
    hash2 = imagehash.whash(Image.fromarray(image2))
    return abs(hash1-hash2) <= 9


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
    id_img = list_id_face[0]

    # Check hash if they are the same picture
    if is_same_image(id_img, selfie_img):
        return error('Found the same images. Please take new picture of you and your id')

    # Check number of face in selfie images
    selfie_face_loc = face_recognition.api.face_locations(selfie_img, number_of_times_to_upsample=2, model='cnn')
    if not selfie_face_loc:
        return error("Can't find any face in your selfie. Please take a new picture of you")
    elif len(selfie_face_loc) > 1:
        return error("Multiple faces have been found. Please take a new picture of only you")
    id_face_loc = face_recognition.api.face_locations(id_img, number_of_times_to_upsample=2, model='cnn')
    if not id_face_loc:
        return error("Can't find any face in the ID. Please take a new picture of your ID")

    # Compare distance
    face1_encoding = face_recognition.face_encodings(id_img, known_face_locations=id_face_loc)[0]
    face2_encoding = face_recognition.face_encodings(selfie_img, known_face_locations=selfie_face_loc)[0]
    face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
    return {
        'distance': face_distance,
        'matched': 'True' if face_distance < 0.6 else 'False',
        'matched_strict': 'True' if face_distance < 0.5 else 'False'
    }


if __name__ == "__main__":
    img = cv2.imread('./test_img/1.png')
    print(get_profile_image_from_layout(img))
