
# Load the model at startup if it exists
if os.path.exists(MODEL_PATH):
    print("Loading ML model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print("WARNING: spam_model.pkl not found! Please ru