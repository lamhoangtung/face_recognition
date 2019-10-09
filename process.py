import base64
import io
import json
import time

import cv2
import imutils
import numpy as np
import requests
from PIL import Image

import face_recognition
import imagehash


def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_image_base64(base64string):
    image = Image.open(io.BytesIO(base64.b64decode(base64string)))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_profile_image_from_layout(id_img):
    layout_api_address = 'http://35.240.219.152:8989/profile_image'
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    _, img_encoded = cv2.imencode('.jpg', id_img)
    response = requests.post(layout_api_address, data=img_encoded.tostring(),
                             headers=headers)
    response = json.loads(response.text)
    if len(response['prediction']) == 0:
        return None
    else:
        most_conf = max(response['prediction'], key=lambda x: x['confidence'])
        return read_image_base64(most_conf['cropped'])


def is_same_image(image1, image2):
    hash1 = imagehash.whash(Image.fromarray(image1))
    hash2 = imagehash.whash(Image.fromarray(image2))
    return abs(hash1-hash2) <= 15


def get_the_biggest_face(face_loc):
    return [max(face_loc, key=lambda x:abs(x[2]-x[0])*abs(x[3]-x[1]))]


def error(message):
    print('--------------------------------------------------------------------------')
    return {
        'success': False,
        'message': message
    }


def process(id_img, selfie_img):
    start_time = time.time()
    print('Start processing ...')
    if isinstance(id_img, bytes):
        id_img = imread_buffer(id_img)
    if isinstance(selfie_img, bytes):
        selfie_img = imread_buffer(selfie_img)

    # Call drake's API to get the cropped profile_image
    print('1. Calling profile image crop API ...')
    id_img = get_profile_image_from_layout(id_img)
    if id_img is None:
        return error("Can't find any face in the ID. Please take a new picture of your ID")
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))

    # Check hash if they are the same picture
    start_time = time.time()
    print('2. Checking duplicate image ...')
    if is_same_image(id_img, selfie_img):
        return error('Found the same images. Please take new picture of you and your id')
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))

    # Check number of face in selfie images
    start_time = time.time()
    print('3. Cropping face from selfie image ...')
    for angle in [0, 90, 270, 360]:
        print('- Trying angle:', angle)
        rotated_selfie = imutils.rotate(selfie_img, angle=angle)
        selfie_face_loc = face_recognition.api.face_locations(rotated_selfie, number_of_times_to_upsample=1)  # , model='cnn')
        if len(selfie_face_loc) > 1:
            print('- Multiple faces have been found. Selecting the biggest ones')
            selfie_face_loc = get_the_biggest_face(selfie_face_loc)
        if len(selfie_face_loc) == 1:
            print('-> Found it!')
            selfie_img = rotated_selfie
            break

    if not selfie_face_loc:
        return error("Can't find any face in your selfie. Please take a new picture of you")
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))

    start_time = time.time()
    print('4. Cropping face from ID image ...')
    id_face_loc = face_recognition.api.face_locations(id_img, number_of_times_to_upsample=1)  # , model='cnn')
    if not id_face_loc:
        return error("Can't find any face in the ID. Please take a new picture of your ID")
    elif len(id_face_loc) > 1:
        print('- Multiple faces have been found. Selecting the biggest ones')
        id_face_loc = get_the_biggest_face(id_face_loc)
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))

    # Compare distance
    start_time = time.time()
    print('5. Dewarp and calculating face embedding ...')
    face1_encoding = face_recognition.face_encodings(id_img, known_face_locations=id_face_loc)[0]
    face2_encoding = face_recognition.face_encodings(selfie_img, known_face_locations=selfie_face_loc)[0]
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))

    start_time = time.time()
    print('6. Getting face distance ...')
    face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
    print('-> Done. Tooks {} secs'.format(time.time()-start_time))
    print('--------------------------------------------------------------------------')
    return {
        'distance': face_distance,
        'matched': 'True' if face_distance < 0.6 else 'False',
        'matched_strict': 'True' if face_distance < 0.5 else 'False'
    }


if __name__ == "__main__":
    img = cv2.imread('./test_img/1.png')
    print(get_profile_image_from_layout(img))
