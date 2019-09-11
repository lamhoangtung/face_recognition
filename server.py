import os

import cv2
import numpy as np
from flask import (Flask, Markup, jsonify, redirect, render_template, request,
                   url_for)
from json2html import json2html
from waitress import serve

import face_recognition

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


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


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/web/predict', methods=['POST'])
def web_predict():
    image1 = request.files.get('image1').read()
    image2 = request.files.get('image2').read()
    result = process(image1, image2)
    result = Markup(json2html.convert(result))
    return render_template('index.html', result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')
    if image1 and image2:
        try:
            image1, image2 = image1.read(), image2.read()
            result = process(image1, image2)
        except Exception as ex:
            result = {'error': str(ex)}
            print(ex)
    elif request.data is not None and isinstance(request.data, list):
        try:
            image1 = cv2.imdecode(np.fromstring(request.data[0], np.uint8),
                                  cv2.IMREAD_COLOR)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.imdecode(np.fromstring(request.data[1], np.uint8),
                                  cv2.IMREAD_COLOR)
            image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            result = process(image1, image2)
        except Exception as ex:
            result = {'error': str(ex)}
            print(ex)
    else:
        ex = 'Please post request.data as list of 2 encoded image as string.'
        result = {'error': ex}
        print(ex)
    print(result)
    # return the data dictionary as a JSON response
    return jsonify(result)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8082)))
