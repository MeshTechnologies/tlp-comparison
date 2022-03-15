import argparse
import tensorflow as tf

from utils.utils import _load_yaml, _dump_yaml
from data.data_provider import DataProvider
from tuner import ModelTuner
from trainer import ModelTrainer

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
        help="path to the desired config yaml file"
    )
    parser.add_argument(
        "hyperparams_path",
        type=str,
        help="path to the desired hyperparameters.yaml output log file"
    )
    parser.add_argument(
        "-t", "--tune",
        action="store_true",
        help="flag to initiate tuning job"
    )
    args = parser.parse_args()

    # Load config
    config = _load_yaml(args.config_path)

    # Set TF seed
    tf.random.set_seed(config["params"]["seed"])

    ####################################################################
    # Load dataset
    ####################################################################
    print("\n" + "#" * 25 + "\nLoading Dataset\n" + "#" * 25 + "\n")

    data_provider = DataProvider(config)

    ####################################################################
    # Hyperparameter tuning
    ####################################################################
    if config['tuning']['tune']:
        print("\n" + "#" * 25 + "\nHyperparameter Tuning\n" + "#" * 25 + "\n")

        tuner = ModelTuner(config).tuner

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs/tune",
            histogram_freq=1,
        )

        tuner.search(
            data_provider.train_ds,
            validation_data=data_provider.val_ds,
            epochs=config["params"]["n_epochs"],
            callbacks=[tensorboard_callback],
        )

        # Insert best hyperparameter settings into model config file as 'value'
        best_hparam_set = tuner.get_best_hyperparameters(num_trials=1)[0].values
    else:
        print("Skipping tuning job. Training hyperparameters will be loaded from the 'default_value' fields in {}".format(args.config_path))
        best_hparam_set = {hparam: config["tuning"]["hyperparams"][hparam]["default_value"] for hparam
                           in config["tuning"]["hyperparams"].keys()}

    # Set hyperparameters values for training
    print("updating {} with hyperparameter values".format(args.config_path))
    _dump_yaml(best_hparam_set, args.hyperparams_path)

    ####################################################################
    # Train Model
    ####################################################################
    print("\n" + "#" * 25 + "\nFit Model\n" + "#" * 25 + "\n")

    # Final hparam set. If tune, then get from best set. Else, get from default values
    hparams = _load_yaml(args.hyperparams_path)

    # Create ModelTrainer
    model = ModelTrainer(config, hparams)

    # Train model
    metrics = model.train(data_provider.train_ds, data_provider.val_ds)

    # Evaluate on test set and log metrics
    test_metrics = model.test(data_provider.test_ds)
    metrics.update(test_metrics)

    # Save tflite model
    model.save_tflite()
