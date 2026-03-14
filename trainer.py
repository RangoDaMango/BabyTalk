import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_features

DATASET = "user_data"
MODEL_PATH = "models/personal_model.pkl"


def retrain():

    X = []
    y = []

    if not os.path.exists(DATASET):
        return

    for label in os.listdir(DATASET):

        folder = os.path.join(DATASET, label)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            try:

                features = extract_features(path)

                X.append(features)
                y.append(label)

            except:
                pass


    if len(X) < 5:
        return


    model = RandomForestClassifier()

    model.fit(np.array(X), np.array(y))

    joblib.dump(model, MODEL_PATH)
