from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
pca = load("pca.joblib")
vqc = load("quantum_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    vector = np.array(data['vector']).reshape(1, -1)
    reduced = pca.transform(vector)
    prediction = vqc.predict(reduced)[0]
    label = "autorise" if prediction == 1 else "refuse"
    return jsonify({"label": label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
