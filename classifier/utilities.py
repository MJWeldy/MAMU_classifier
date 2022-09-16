#functions.py

#CLASSIFIER_PATH = 'BirdNET/checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite'
import tensorflow as tf


def load_model(path):
    classifier = tf.keras.models.load_model(path)
    return classifier

def generate_class_map(FG_FOLDER):
    import os
    dirnames = os.listdir(FG_FOLDER)
    dirnames.sort() 
    class_map = {}
    for i, dirname in enumerate(dirnames):
        class_map[dirname] = i
    return class_map