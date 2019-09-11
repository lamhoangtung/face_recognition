import os
from importlib import import_module

import numpy as np

import cv2
from flask import (Flask, Markup, jsonify, redirect, render_template, request,
                   url_for)
from json2html import json2html
from waitress import serve

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/web/predict', methods=['POST'])
def web_predict():
    images = request.files.getlist('image')
    images = [image.read() for image in images]
    result = process(images)
    result = Markup(json2html.convert(result))
    return render_template('index.html', result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    images = request.files.getlist("image")
    if images:
        try:
            images = [image.read() for image in images]
            result = process(images)
        except Exception as ex:
            result = {'error': str(ex)}
            print(ex)
    elif request.data is not None:
        try:
            image = cv2.imdecode(np.fromstring(request.data, np.uint8),
                                 cv2.IMREAD_COLOR)
            # image = Image.open(io.BytesIO(request.files["image"].read()))
            result = process(image)
        except Exception as ex:
            result = {'error': str(ex)}
            print(ex)
    print(result)
    # return the data dictionary as a JSON response
    return jsonify(result)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
