import tensorflow as tf
import numpy as np
from flask_socketio import emit
from .. import socketio
import threading

training_thread = None
testing_thread = None
stop_training_flag = threading.Event()
stop_testing_flag = threading.Event()

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def train_model(num_processors):
    global stop_training_flag
    num_processors = int(num_processors)
    strategy = tf.distribute.MirroredStrategy(devices=[f'/cpu:{i}' for i in range(num_processors)])
    with strategy.scope():
        model = create_model()
        (x_train, y_train), _ = load_data()
        for epoch in range(5):
            if stop_training_flag.is_set():
                socketio.emit('training_log', {'msg': 'Training stopped by user'}, namespace='/mnist')
                break
            model.fit(x_train, y_train, epochs=1, verbose=2, callbacks=[TrainingLogger(epoch)])

class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, epoch):
        self.epoch = epoch
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {self.epoch + 1}, Loss: {logs.get('loss'):.4f}, Accuracy: {logs.get('accuracy'):.4f}"
        socketio.emit('training_log', {'msg': msg}, namespace='/mnist')

def test_model(num_processors):
    global stop_testing_flag
    num_processors = int(num_processors)
    strategy = tf.distribute.MirroredStrategy(devices=[f'/cpu:{i}' for i in range(num_processors)])
    with strategy.scope():
        model = create_model()
        _, (x_test, y_test) = load_data()
        for i in range(1):
            if stop_testing_flag.is_set():
                socketio.emit('testing_log', {'msg': 'Testing stopped by user'}, namespace='/mnist')
                break
            results = model.evaluate(x_test, y_test, verbose=2)
            msg = f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}"
            socketio.emit('testing_log', {'msg': msg}, namespace='/mnist')

@socketio.on('start_training', namespace='/mnist')
def handle_start_training(message):
    global training_thread, stop_training_flag
    num_processors = message['num_processors']
    stop_training_flag.clear()
    training_thread = threading.Thread(target=train_model, args=(num_processors,))
    training_thread.start()

@socketio.on('stop_training', namespace='/mnist')
def handle_stop_training():
    global stop_training_flag
    stop_training_flag.set()

@socketio.on('start_testing', namespace='/mnist')
def handle_start_testing(message):
    global testing_thread, stop_testing_flag
    num_processors = message['num_processors']
    stop_testing_flag.clear()
    testing_thread = threading.Thread(target=test_model, args=(num_processors,))
    testing_thread.start()

@socketio.on('stop_testing', namespace='/mnist')
def handle_stop_testing():
    global stop_testing_flag
    stop_testing_flag.set()
