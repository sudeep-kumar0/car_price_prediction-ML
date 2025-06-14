from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'LinearRegressionModel.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise Exception(f"Model could not be loaded: {e}")

# Load car data
data_path = os.path.join(os.path.dirname(__file__), 'Cleaned_Car_data.csv')
try:
    car = pd.read_csv(data_path)
except Exception as e:
    raise Exception(f"Data could not be loaded: {e}")

# Prepare dropdown values
companies = sorted(car['company'].unique())
car_models = car[['company', 'name']].drop_duplicates()
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

# Map company to models
company_model_map = defaultdict(list)
for _, row in car_models.iterrows():
    company_model_map[row['company']].append(row['name'])

@app.route('/')
def index():
    return render_template('index.html',
                           companies=companies,
                           years=years,
                           fuel_types=fuel_types,
                           company_models=company_model_map)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form['company']
        car_model = request.form['car_models']
        year = int(request.form['year'])
        fuel_type = request.form['fuel_type']
        kms_driven = int(request.form['kilo_driven'])

        # Prepare input dataframe
        input_df = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Predict and return price
        prediction = model.predict(input_df)
        predicted_price = np.round(prediction[0], 2)
        return str(predicted_price)

    except Exception as e:
        return str(f"Prediction error: {e}")

# Ensure this only runs locally
if __name__ == '__main__':
    app.run(debug=True)

