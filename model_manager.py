import joblib
import os

GLOBAL_MODEL = "models/global_model.pkl"
PERSONAL_MODEL = "models/personal_model.pkl"


def load_model(path):

    if os.path.exists(path):
        return joblib.load(path)

    return None


def predict(features):

    global_model = load_model(GLOBAL_MODEL)
    personal_model = load_model(PERSONAL_MODEL)

    predictions = {}

    if global_model:

        probs = global_model.predict_proba([features])[0]
        classes = global_model.classes_

        for c, p in zip(classes, probs):
            predictions[c] = predictions.get(c, 0) + p * 0.7


    if personal_model:

        probs = personal_model.predict_proba([features])[0]
        classes = personal_model.classes_

        for c, p in zip(classes, probs):
            predictions[c] = predictions.get(c, 0) + p * 0.3


    if not predictions:
        return "unknown"

    return max(predictions, key=predictions.get)
