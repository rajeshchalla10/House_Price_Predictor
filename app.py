from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle
import numpy as np
import math


app = Flask(__name__)
data = pd.read_csv('Bengaluru_House_Cleaned_Data.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')

def index():
    
   
    locations = sorted(data['location'].unique())
    area_type = sorted(data['area_type'].unique())

    return render_template('index_test.html',locations=locations,area_type=area_type)

@app.route('/predict', methods=['GET','POST'])
def predict():

    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    area_type = request.form.get('area_type')
    balcony = request.form.get('balcony')

#    print(location,bhk,bath,sqft,area_type,balcony)

    input = pd.DataFrame([[location,sqft,bath,bhk,area_type,balcony]],columns=['location','total_sqft','bath','bhk','area_type', 'balcony'])
    prediction = pipe.predict(input)[0] * 1e5
    prediction_indian_system = indian_system(prediction)

    #np.round(prediction,2)

    return str(prediction_indian_system)




def indian_system(value):
    
    
    """
    Formats a numerical value into a string representing Indian currency format
    by breaking it down into Crores and Lakhs only, discarding anything less than a Lakh.

    Args:
        value (float or int): The number to format.

    Returns:
        str: The formatted string in Indian currency format breakdown (rounded down to Lakhs).
    """
    if not isinstance(value, (int, float)):
        raise TypeError("Input value must be a number.")

    # Handle zero and near-zero values - adjust threshold as we're rounding to Lakhs
    if abs(value) < 100000: # Consider values less than 1 Lakh as effectively zero
        return '0'

    value = float(value)
    is_negative = value < 0
    abs_value = abs(value) # Work with the absolute value for breakdown

    # Define the units and their values in descending order
    units = [
        (1e7, 'Crore'),
        (1e5, 'Lakh'),
        # No units for less than a Lakh as we discard it
    ]

    parts = []
    # Work with the integer part of the absolute value
    remaining_value = int(abs_value)

    for threshold, unit in units:
        if remaining_value >= threshold:
            # Calculate the number of full units
            num_units = math.floor(remaining_value / threshold)
            if num_units > 0:
                # Add the unit part to the list
                parts.append(f"{int(num_units)} {unit}{'s' if num_units > 1 and unit else ''}")
                # Subtract the value of these units from the remaining value
                remaining_value -= num_units * threshold

    # Handle the remaining integer value less than a Lakh (discarded)
    # This will always be less than 100000 because the loop stops at 1e5.
    # We only add a 'Lakhs' value if no larger units were added and the original absolute integer value >= 100000.
    if not parts and int(abs_value) >= 100000:
        # Calculate the number of full lakhs and add it
        num_lakhs = math.floor(int(abs_value) / 100000)
        if num_lakhs > 0:
            parts.append(f"{int(num_lakhs)} Lakh{'s' if num_lakhs > 1 else ''}")


    formatted_string = " ".join(parts).strip()

    if not formatted_string or formatted_string == '0':
        # This handles cases where the number is less than 1 Lakh
        return '0'

    # Add the negative sign at the beginning of the formatted string if the original value was negative
    if is_negative:
        return "-" + formatted_string
    return formatted_string






if __name__ == '__main__':
    app.run(debug=True, port= 5500)
