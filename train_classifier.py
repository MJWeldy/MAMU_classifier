import scaper
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow import lite as tflite

from BirdNET import config as cfg
from BirdNET import analyze
from BirdNET import model
from BirdNET import embeddings

from classifier import utilities

FG_FOLDER = "foreground/"
BG_FOLDER = "background/"


event_template = {
    'label': ('choose', []),
    'source_file': ('choose', []),
    'source_time': ('uniform', 0, 0.3),
    'event_time': ('uniform', 0.1, 2.3),
    'event_duration': ('uniform', 0.3, 10),
    'snr': ('uniform', -5, 3),
    'pitch_shift': ('uniform', -1, 1),
    'time_stretch': ('uniform', 0.9, 1.0)
}

def generate(duration, fg_folder, bg_folder, event_template, seed):
    sc = scaper.Scaper(
        duration=duration,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    sc.sr = 48000
    sc.n_channels = 1
    sc.ref_db = np.random.randint(-40,-15)
    #print(sc.ref_db)
    sc.add_background(
        label=('choose', ['Nice_weather', 'Rain', 'River_stream', 'Wind']),
        #label=('const', 'HJA'),
        source_file=("choose", []),
        source_time=("uniform", 0, 20),
    )
    min_events = 1
    max_events = 5
    n_events = np.random.randint(min_events, max_events + 1)
    for _ in range(n_events):
        sc.add_event(**event_template)
    return sc.generate(fix_clipping=True, allow_repeated_label=False, allow_repeated_source=False,disable_sox_warnings=True);


def create_batch(batch_size, seed):
    sounds = []
    labels = []
    for i in range(batch_size):
        mixture_audio, _, annotation_list, _ = generate(3.0, fg_folder, bg_folder, event_template, seed = seed*i)
        sounds.append(np.squeeze(mixture_audio.astype(np.float32)))
        labels.append(annotation_list)
    features = model.embeddings(sounds)
    label_array = vectorize_labels(labels, class_map)
    return features, label_array

def unpack_labels(labels_element):
    names = []
    for element in labels_element:
        _, _, name = element
        names.append(name)
    return names
    
def vectorize_labels(labels, class_map):
    results = np.zeros((len(labels), len(class_map)))
    for i, label_set in enumerate(labels):
        label_list = unpack_labels(label_set)
        for j in label_list:
            if j in class_map.keys():
                col = class_map[j]
                results[i, col] = 1
    return results

def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()

# Tensorflow compiled functions
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = my_model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, my_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_weights))

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()
    return logs

@tf.function
def test_step(inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["tes_" + metric.name] = metric.result()

    loss_tracking_metric.update_state(loss)
    logs["test_loss"] = loss_tracking_metric.result()
    return logs

def main():

    class_map = utilities.generate_class_map(FG_FOLDER)
    n_classes = len(class_map.keys())

    classifier_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(320), dtype=tf.float32,
                            name='input_embedding'),
            #tf.keras.layers.Dense(16, activation='relu'),
            #tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(n_classes, activation = 'sigmoid')
    ], name='classifier_model') 

    loss_fn = keras.losses.BinaryCrossentropy() 
    optimizer = keras.optimizers.Adam(learning_rate=1e-3) 
    metrics = [keras.metrics.BinaryAccuracy()] 
    loss_tracking_metric = keras.metrics.Mean() 

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    test_batch, test_targets = create_batch(batch_size = 1000, seed = 5000000)

    for sim in range(SIMULATION_STEPS):
        reset_metrics()
        input_batch, targets_batch = create_batch(batch_size = BATCH_SIZE, seed = sim)
        train_logs = train_step(input_batch, targets_batch)
        train_loss.append(train_logs['loss'])
        train_acc.append(train_logs['binary_accuracy'])

        if(sim % 10 != 0):
            #input_batch, targets_batch = create_batch(batch_size = BATCH_SIZE*5, seed = 5000000)
            test_logs = test_step(test_batch, test_targets)
            test_loss.append(test_logs['loss'])
            test_acc.append(test_logs['binary_accuracy'])
            print(f"\nOn simulation {sim}: 
                    \ntraining loss: { train_logs['loss'] } 
                    \ntraining accuracy: { train_logs['binary_accuracy'] } 
                    \nvalidation loss: { test_logs['test_loss'] } 
                    \nvalidation accuracy: { test_logs['test_accuracy'] }")
    
    classifier_model.save(f'./classifier/checkpoints/{VERSION_NUMBER}.h5')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_steps')
    parser.add_argument('batch_size')
    parser.add_argument('version_number')
    args = parser.parse_args()
    SIMULATION_STEPS = args.simulation_steps
    BATCH_SIZE = args.batch_size
    VERSION_NUMBER = args.version_number
    MODEL_SAVE_PATH = args.model_save_path
    main()