from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import pickle
import xgboost
from keras.models import load_model
from keras.preprocessing import image
from sklearn.ensemble import _forest

model = load_model('Braintumour.h5')
pneumoniamodel = load_model('Pneumonia_model.h5')
model3 = pickle.load(open('bcancer.pkl', 'rb'))
diabeticsmodel = pickle.load(open('Diabetes.pkl', 'rb'))
malariamodel= load_model('Malaria_model1.h5')
strokemodel = pickle.load(open('stroke.pkl', 'rb'))
kidneymodel = pickle.load(open('kidney (2).pkl', 'rb'))
livermodel = pickle.load(open('liver_model.pkl', 'rb'))


COUNT = 0
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/braintumor1')
def braintumor():
    return render_template('braintumor.html')

@app.route('/pneumonia1')
def pneumoniapage():
    return render_template('pneumonia.html')

@app.route('/bcancer')
def bcancer():
    return render_template('bcancer.html')

@app.route('/malaria')
def malaria():
    return render_template('malaria.html')

@app.route('/diabetics')
def diabetics():
    return render_template('diabetics.html')

@app.route('/stroke')
def stroke():
    return render_template('stroke.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

#brain tumor
@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('prediction.html', data=preds)

@app.route('/home-pneumonia', methods=['POST'])
def homepneumonia():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = pneumoniamodel.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('prediction-pneumonia.html', data=preds)

@app.route('/predict3', methods=['POST', 'GET'])
def predict3():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction3 = model3.predict(final3)

    if prediction3 == 0:
        return render_template('bcancer.html', pred='The cancer is malignant. Consult a Oncologist')
    else:
        return render_template('bcancer.html', pred='The cancer is benign, Stay Safe!!!')

@app.route('/predictdia',methods=['POST' , 'GET'])
def predictdia():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = diabeticsmodel.predict(final)

    if prediction == 0:
        return render_template('diabetics.html', pred='The person is non diabetic. Stay safe!!!')
    else:
        return render_template('diabetics.html', pred='The person is diabetic, Go and consult a Endocrinologist.')

@app.route('/home-malaria', methods=['POST'])
def homemalaria():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = malariamodel.predict(img_arr)
    COUNT += 1
    return render_template('malariaprediction.html', data=prediction)

@app.route('/predictliver', methods=['POST', 'GET'])
def predictliver():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction3 = livermodel.predict(final3)

    if prediction3 == 1:
        return render_template('liver.html', pred='The person is suffering with Liver Disease. Please consult  hepatologists')
    else:
        return render_template('liver.html', pred='No Liver Disease. Stay Safe!!!')

@app.route('/predictstroke', methods=['POST', 'GET'])
def predictstroke():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction3 = strokemodel.predict(final3)

    if prediction3 == 0:
        return render_template('stroke.html', pred='The person is Safe.!!!')
    else:
        return render_template('stroke.html', pred='The person may suffer from stroke, Please consult a Cardiologist Soon.')

@app.route('/predictkidney', methods=['POST', 'GET'])
def predictkidney():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction3 = kidneymodel.predict(final3)

    if prediction3 == 0:
        return render_template('kidney.html', pred='The person doesnt have kidney disease')
    else:
        return render_template('kidney.html', pred='The person may suffer or suffering from Kidney Disease, So consult a Urologist')

if __name__ == '__main__':
    app.run(debug=True)

