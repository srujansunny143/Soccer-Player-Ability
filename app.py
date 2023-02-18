import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))
@app.route('/')
def home():
    return render_template('/home.html')

@app.route('/Soccer Player Ability predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Current Ability is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)