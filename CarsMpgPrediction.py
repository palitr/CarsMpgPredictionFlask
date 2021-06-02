from flask import Flask, request, jsonify, url_for, render_template
from tensorflow.keras.models import load_model
import numpy as np
import requests
import os


app = Flask(__name__)

model_ld = load_model(os.path.join(os.getcwd(), 'mpg_model.h5'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    
    array = np.zeros((1,7))
    for idx, val in enumerate(int_features):
        array[0,idx-1] = val

    prediction = model_ld.predict(array)
    
    output = np.round(float(prediction[0]))

    return render_template('index.html', prediction_text='MPG should be: {}'.format(output))
    

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model_ld.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False) 
