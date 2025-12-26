from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## importing ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# creating a home route
@app.route("/")
def index():
    return render_template('index.html')

# creating a preedicting route where the values will entered in the frontend
@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method == 'POST':
        pass
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True,port=8004)