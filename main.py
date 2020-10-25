import flask
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import imageio
import scipy.misc
from PIL import Image
from flask_restful import Resource, Api
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)
f=open("model.pkl","rb")
model = pickle.load(f)
f.close()
#메인페이지 라우팅
@app.route("/")
@app.route("/index")

def index():
    return flask.render_template('test.html')

#데이터 예측 처리
@app.route('/predict', methods=['POST', 'GET'])
def make_prediction():
    data = {"success":False}
    if request.method == 'POST':
        file = request.files.get('image','')

        if not file : return jsonify({'result': 'file not found'})

        img = imageio.imread(file)
        img = img[:, :, :3]
        img_copy = np.resize(img, 28*28)

        img_copy = img_copy.reshape(1, -1)

        prediction = model.predict(img_copy)
        label = str(np.squeeze(prediction))

        if label =='10': label = '0'
        
        return jsonify({'class_id': label})
    return jsonify({'result': 'failed: wrong method'})


if __name__ == '__main__':
    f=open("model.pkl","rb")
    model = pickle.load(f)
    f.close()
    app.run(host='0.0.0.0', port = 8000, debug = True)
