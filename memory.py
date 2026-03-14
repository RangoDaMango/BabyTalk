import os
import shutil
from trainer import retrain

DATASET = "user_data"

def store_sample(file, label):

    folder = os.path.join(DATASET, label)

    os.makedirs(folder, exist_ok=True)

    new_path = os.path.join(folder, os.path.basename(file))

    shutil.move(file, new_path)

    retrain()
