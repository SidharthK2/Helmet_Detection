from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Img_utils.utils import decodeImage
from detect import Predictor
from loggerClass import getLog

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

logger = getLog('detect.py')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.obj_detect = Predictor()


@app.route("/")
def home():
    try:
        return render_template("index.html")
        logger.info('Homepage loaded')
    except Exception as e:
        logger.info('could not load homepage')
        print(f'error {e}')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        logger.info('Image received')
        decodeImage(image, clApp.filename)
        logger.info('image decoded')
        result = clApp.obj_detect.run_inference()
        logger.info('Inference run')
        return jsonify(result)
        logger.info('result returned')
    except Exception as e:
        logger.info('could not load results')
        print(f'error {e}')


if __name__ == "__main__":
    clApp = ClientApp()
    port = 8000
    app.run(host='127.0.0.1', port=port)
