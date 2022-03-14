from model import get_model

import argparse
import tensorflow as tf
from dvclive.keras import DvcLiveCallback


def train(params):
    print(tf.__version__)

    model = get_model(params.hidden_dim, params.dropout_p, params.output_dim, params.learning_rate)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="output/logs/train",
        histogram_freq=1,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least "early_stopping_threshold" epochs"
        patience=threshold,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='output/ckpt/',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    history = model.fit(
        dataset,
        validation_data=dataset,
        epochs=params.epochs,
        callbacks=[
            tensorboard_callback,
            early_stopping_callback,
            model_checkpoint_callback,
            DvcLiveCallback()
        ]
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('output/model.tflite', "wb", ) as f:
        f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout_p', type=float)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    train(args)
