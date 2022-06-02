#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
predictModel = pickle.load(open("admissionModel","rb"))
scaleModel = pickle.load(open("scaleAdmissionModel","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [float(x) for x in request.form.values()]
    scaledValue = scaleModel.transform([int_features]) 
    result = predictModel.predict(scaledValue)    
    return render_template('index.html', pred='Your chances of Admission Is',result[0],'%')   
if __name__ == '__main__':
    app.run(debug=False)
