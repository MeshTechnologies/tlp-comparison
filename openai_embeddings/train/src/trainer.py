import tensorflow as tf
from tensorflow.keras import losses
from dvclive.keras import DvcLiveCallback

from models.model import build_linear_classifier


class ModelTrainer:
    """ Base class for models with training, evaluation, and saving.

    Attributes:
        config (dict): The configuration for the train/tune run.
        params (dict): The non-tunable parameters
        model (tf.keras.Model): The model to train.

    """

    def __init__(self, config, hparams):
        self.config = config
        self.params = config["params"]
        self.model = generate_model(config, hparams)

    def train(self, train_ds, val_ds) -> dict:
        """ Train the model

        Args:
            train_ds (tf.data.Dataset): The training dataset
            val_ds (tf.data.Dataset): The validation dataset

        Returns:
            dict: The metrics for the training run

        """

        # Print the model summary
        self.model.summary()

        # Create the callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor=self.params["objective"],
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least "early_stopping_threshold" epochs"
            patience=self.params["early_stopping_threshold"],
            verbose=1,
            restore_best_weights=True
        )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config["output_paths"]["model"]["keras_model"],
            monitor=self.params["objective"],
            verbose=0,

            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs/train",
            histogram_freq=1,
        )

        # Train the model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.params["n_epochs"],
            callbacks=[
                model_checkpoint_callback,
                early_stopping_callback,
                DvcLiveCallback(),
                tensorboard_callback
            ],
        )

        # Return the metrics
        return {metric: history.history[metric][-1] for metric in history.history.keys()}

    def save_tflite(self):
        """ Convert the model to tflite format and save it to the output path. """
        # Convert the model to tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        # Save the model to the output path
        with open(self.config["output_paths"]["model"]["tflite_model"], "wb", ) as f:
            f.write(tflite_model)

    def test(self, test_ds) -> dict:
        """ Evaluate the model on the test dataset.

        Args:
            test_ds (tf.data.Dataset): The test dataset

        Returns:
            dict: The metrics for the evaluation run

        """
        return self.model.evaluate(
            test_ds,
            batch_size=self.params["batch_size"],
            return_dict=True
        )


def generate_model(config, hparams) -> tf.keras.Model:
    """ Generate a model based on the params, hyperparameters, and models in src/models/.

    This function can also be used in the Keras Tuner.

    Args:
        config (dict): The configuration for the train/tune run.
        hparams (dict): The hyperparameters for the model.

    Returns:
        tf.keras.Model: The compiled Keras model.

    """

    model = None

    if hparams["model_architecture"] == "Linear_classifier":
        model = build_linear_classifier(config, hparams)
    else:
        raise ValueError("Invalid model architecture '{}'. Amend config file.".format(hparams["model_architecture"]))

    if hparams["loss"] == "Crossentropy":
        if config["params"]["output_dim"] > 2:
            loss = losses.SparseCategoricalCrossentropy(True)
        else:
            loss = losses.BinaryCrossentropy(True)
    else:
        raise ValueError("Invalid loss function '{}'. Amend config file.".format(hparams["loss"]))

    if hparams["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"], )
    else:
        raise ValueError("Invalid optimizer '{}'. Amend config file.".format(hparams["optimizer"]))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
    )
    return model
