# How to Use the 'models' Module
This module should contain scripts with functions that return uncompiled Keras models to be used in the generate_model() 
function in the 'trainer.py' module.

e.g.
**`lstm.py`**
```python
import tensorflow as tf
from tensorflow import keras


def build_vanilla_lstm(config, hparams) -> keras.Model:
    """ Builds the uncompiled Keras LSTM classifier model.

    Args:
        config (dict): The configuration parameters.
        hparams (dict): The model hyperparameters.

    Returns:
        keras.Model: The uncompiled Keras model.

    """
    params = config["params"]

    model = keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            params["vocab_size"],
            hparams["embedding_dim"],
            input_length=params["sequence_len"],
            mask_zero=True
        )
    )

    model.add(tf.keras.layers.Dropout(hparams["dropout_p"]))

    if hparams['bidirectional']:
        model.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hparams["hidden_dim"])))
    else:
        model.add(tf.keras.layers.LSTM(hparams["hidden_dim"]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(hparams["dropout_p"]))
    model.add(tf.keras.layers.Dense(hparams["hidden_dim"], activation='relu'))
    model.add(tf.keras.layers.Dropout(hparams["dropout_p"]))
    model.add(tf.keras.layers.Dense(params["output_dim"], name="output"))

    return model
```