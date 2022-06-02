#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
predictModel = pickle.load(open("Stress_model_save (1)","rb"))
scaleModel = pickle.load(open("scaleModel","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [float(x) for x in request.form.values()]
    scaledValue = scaleModel.transform([int_features]) 
    loadModel = pickle.load(open("Stress_model_save (1)","rb"))
    result = loadModel.predict(scaledValue)    
    if result[0]==0:
        return render_template('index.html', pred='Your Stress level is Normal or Low')
    elif result[0]==1:
        return render_template('index.html', pred='Your Stress level is Medium low')
    elif result[0]==2:
        return render_template('index.html', pred='Your Stress level is Medium')
    elif result[0]==3:
        return render_template('index.html', pred='Your Stress level is Medium High')   
    elif result[0]==4:
        return render_template('index.html', pred='Your Stress level is High')   
if __name__ == '__main__':
    app.run(debug=False)
