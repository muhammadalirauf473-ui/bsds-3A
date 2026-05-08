from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# ==============================
# CREATE FLASK APP
# ==============================
app = Flask(__name__)

# ==============================
# LOAD TRAINED MODEL
# ==============================
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Iris class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

class_descriptions = {
    'Setosa': 'Small, delicate flowers with narrow petals.',
    'Versicolor': 'Medium-sized flowers commonly found in North America.',
    'Virginica': 'Large flowers with wide petals.'
}

class_colors = {
    'Setosa': '#4ade80',
    'Versicolor': '#60a5fa',
    'Virginica': '#f472b6'
}

# ==============================
# HOME ROUTE
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

# ==============================
# PREDICT ROUTE
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Convert to DataFrame
        features = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=[
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ]
        )

        # Prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_class = class_names[prediction]
        confidence = round(probabilities[prediction] * 100, 2)

        description = class_descriptions[predicted_class]
        color = class_colors[predicted_class]

        prob_data = {
            class_names[i]: round(probabilities[i] * 100, 2)
            for i in range(3)
        }

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=confidence,
            description=description,
            color=color,
            prob_data=prob_data,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

# ==============================
# RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=True)