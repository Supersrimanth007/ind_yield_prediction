from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('ind_yield.pkl', 'rb') as file:
    Model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['Average_Rain_Fall_mm_per_year'],
        data['Pesticides_Tonnes'],
        data['Avg_Temp']
    ]
    # One hot encoding for Item
    item_dict = {
        'Cassava': 0, 'Maize': 1, 'Potatoes': 2, 'Rice, paddy': 3,
        'Sorghum': 4, 'Soybeans': 5, 'Sweet potatoes': 6, 'Wheat': 7
    }
    item_encoded = [0] * 8
    item_encoded[item_dict[data['Item']]] = 1
    features.extend(item_encoded)

    prediction = Model.predict([features])[0]
    return jsonify({'predicted_yield': prediction})

if __name__ == '__main__':
    app.run(debug=True)
