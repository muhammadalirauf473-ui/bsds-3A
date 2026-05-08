import os
import joblib
import sklearn
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Path to our trained ML model
MODEL_PATH = 'spam_model.pkl'

# Load the model at startup if it exists
if os.path.exists(MODEL_PATH):
    print("Loading ML model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print("WARNING: spam_model.pkl not found! Please run 'python train_model.py' first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_spam():
    if not model:
        return jsonify({
            "error": "The Machine Learning model is not trained yet. Please run 'python train_model.py' to generate 'spam_model.pkl'."
        }), 500

    data = request.json
    if not data or 'email_content' not in data:
        return jsonify({"error": "No email content provided"}), 400

    email_content = data['email_content']
    
    try:
        # Predict the class (0 = ham, 1 = spam)
        prediction = model.predict([email_content])[0]
        
        # Get probability estimation (confidence)
        probabilities = model.predict_proba([email_content])[0]
        
        # probabilities array format depends on classes ordering, but generally [prob_0, prob_1]
        spam_probability = probabilities[1]
        
        classification = "spam" if prediction == 1 else "not_spam"
        confidence = float(spam_probability) if classification == "spam" else float(probabilities[0])

        # Generate a hardcoded reasoning since traditional ML doesn't natively "explain" itself like an LLM.
        # We can simulate the output to keep our frontend intact.
        if classification == "spam":
            reasoning = f"The Machine Learning model detected strong indicators of spam (keywords, structural patterns) matching trained spam datasets. (Spam Probability: {spam_probability*100:.1f}%)"
        else:
            reasoning = f"The text appears safe and mostly lacks the statistical features associated with spam messages. (Safe Probability: {confidence*100:.1f}%)"

        result = {
            "classification": classification,
            "confidence_score": confidence,
            "reasoning": reasoning
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "error": "An error occurred while analyzing the text with the ML model."
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
