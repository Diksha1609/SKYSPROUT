from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np
import pickle
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

# Load models and scalers (replace with actual paths)
daily_model = pickle.load(open('models/random_forest_model_no_sun_hours.pkl', 'rb'))
hourly_model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
daily_scaler = pickle.load(open('models/scaler_no_sun_hours.pkl', 'rb'))
hourly_scaler = pickle.load(open('models/scaler3.pkl', 'rb'))


with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/scaler3.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Route to render the index.html first
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')

# Route to render the Predictor page
@app.route('/predictor')
def predictor():
    return render_template('Predictor.html')

from flask import Flask, request, jsonify
import numpy as np

@app.route('/predict2', methods=['POST'])
def daily_predict():
    try:
        # Extract form data
        maxtemp = float(request.form['maxtemp'])
        mintemp = float(request.form['mintemp'])
        uvindex = float(request.form['uvindex'])
        dewpoint = float(request.form['dewpoint'])
        windgust = float(request.form['windgust'])
        cloudcover = float(request.form['cloudcover'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        visibility = float(request.form['visibility'])
        windspeedKmph = float(request.form['windspeedKmph'])
        
        # Create the input feature array for the model
        features = np.array([[maxtemp, mintemp, uvindex, dewpoint, windgust, cloudcover,
                              humidity, pressure, visibility, windspeedKmph]])

        # Scale the features for the daily model
        scaled_features = daily_scaler.transform(features)

        # Predict using the daily model
        prediction = daily_model.predict(scaled_features)

        # Get the predicted rainfall value
        rainfall_prediction = prediction[0]  # Adjust if your model returns a different format

        # Return the result as a JSON response
        return jsonify({'predicted_rainfall': rainfall_prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return error message as JSON


@app.route('/predict', methods=['POST'])
def hourly_predict():
    try:
        # Extract form data
        cloudcover = float(request.form['cloudcoverhr'])
        dewpoint = float(request.form['DewpointChr'])
        maxtemp = float(request.form['maxtempChr'])
        sunhour = float(request.form['sunHourhr'])
        humidity = float(request.form['humidityhr'])
        precipmm = float(request.form['precipMMhr'])
        previousDayPrecipMM = float(request.form['previousDayPrecipMM'])
        previousDayMaxTemp = float(request.form['previousDayMaxTemp'])
        twoDaysBeforeMaxTemp = float(request.form['twoDaysBeforeMaxTemp'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        
        # Create the input feature array for the hourly model
        features = np.array([[cloudcover, dewpoint, maxtemp, sunhour, humidity, precipmm, 
                              previousDayPrecipMM, previousDayMaxTemp, twoDaysBeforeMaxTemp, month, day]])

        # Scale the features for the hourly model
        scaled_features = hourly_scaler.transform(features)

        # Predict using the hourly model
        prediction = hourly_model.predict(scaled_features)

        # Process prediction result (e.g., format it as needed)
        hourly_prediction_result = prediction[0]  # or customize the result format if necessary

        # Return the result to the template
        return jsonify({'predicted_rainfall': hourly_prediction_result})


    except Exception as e:
        return str(e)



# Endpoint to serve JSON data
@app.route('/api/data')
def get_data():
    try:
        with open('static/rainfall_data.json', 'r') as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve the index HTML page
@app.route('/charts')
def index_charts():
    return render_template('index_charts.html')


@app.route('/hourly_prediction')
def hourly_prediction():
    return render_template('index_hr.html')

@app.route('/predict22', methods=['POST'])
def predict22():
    data = request.json

    # Extract input values
    cloudcover = data['cloudcover']
    DewPointC = data['DewPointC']
    maxtempC = data['maxtempC']
    sunHour = data['sunHour']
    humidity = data['humidity']
    prev_precipMM = data['prev_precipMM']
    prev_precipMM_2 = data['prev_precipMM_2']
    prev_maxtempC = data['prev_maxtempC']
    prev_maxtempC_2 = data['prev_maxtempC_2']
    month = data['month']
    day = data['day']

    # Generate lag and seasonal features
    month_sin = math.sin(2 * math.pi * month / 12)
    day_cos = math.cos(2 * math.pi * day / 31)

    input_features = [
        cloudcover, DewPointC, maxtempC, sunHour,
        prev_precipMM, prev_precipMM_2, prev_maxtempC, prev_maxtempC_2,
        humidity, humidity, month_sin, day_cos
    ]

    input_df = pd.DataFrame([input_features], columns=[
        'cloudcover', 'DewPointC', 'maxtempC', 'sunHour',
        'precipMM_lag_1', 'precipMM_lag_2', 'maxtempC_lag_1', 'maxtempC_lag_2',
        'humidity_lag_1', 'humidity_lag_2', 'month_sin', 'day_cos'
    ])

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    return jsonify({'prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)






