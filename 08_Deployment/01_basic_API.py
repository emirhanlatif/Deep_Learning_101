from flask import Flask,request,jsonify
import numpy as np
from keras.models import load_model
import joblib

def return_prediction(model, scaler, sample_json):

    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])

    preds_prob = model.predict(flower) 
    class_ind = np.argmax(preds_prob, axis=1)[0]
    return classes[class_ind]



app = Flask(__name__)

@app.route("/")
def index():
    return '<h1>FLASK APP IS RUNNING</h1>'

flower_model=load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route('/api/flower',methods = ['POST'])
def flower_prediction():
    content = request.json
    result = return_prediction(flower_model,flower_scaler,content)
    return jsonify(result)



if __name__=='__main__':
    app.run()