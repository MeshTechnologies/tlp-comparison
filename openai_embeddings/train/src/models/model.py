import tensorflow as tf


def build_linear_classifier(config, hparams) -> tf.keras.Model:
    """ Builds the uncompiled Keras LSTM classifier model.

    Args:
        config (dict): The configuration parameters.
        hparams (dict): The model hyperparameters.

    Returns:
        keras.Model: The uncompiled Keras model.

    """
    params = config["params"]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(hparams["hidden_dim"]),
        tf.keras.layers.Dense(hparams["hidden_dim"], activation='relu'),
        tf.keras.layers.Dropout(hparams["dropout_p"]),
        tf.keras.layers.Dense(params["output_dim"], name='output')
    ])

    return model
