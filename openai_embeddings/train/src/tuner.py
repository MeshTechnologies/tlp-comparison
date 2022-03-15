from types import SimpleNamespace
import tensorflow as tf
import keras_tuner as kt

from trainer import generate_model


class ModelTuner:
    """ Base class for a tuning job using Keras Tuner

    Attributes:
        config (dict): The configuration for the train/tune run.

    """

    def __init__(self, config):
        self.config = config
        self.hparams = config["tuning"]["hyperparams"]
        self.tuner = self.build_tuner()

    def get_tunable_hparams(self, hp=None):
        """ Provides the parsing logic so that hyperparameter tuning can be configured entirely from the config.yaml file.

        Args:
            hp (kt.HyperParameters, optional): Keras Tuner hyperparameter object.
                                               https://keras.io/api/keras_tuner/hyperparameters/

        Returns:
            tunable_hparams (dict): Flattened {hyperparameter:value,...} dict

        """
        tunable_hparams = {}

        if all(self.config["tuning"]["tune"] == False for hparam in self.hparams.keys()):
            raise ValueError("No tunable parameters, set tune to True for at least one hyperparameter")

        for hparam in self.hparams.keys():
            hparams_dict = self.hparams[hparam]
            tuning_config = SimpleNamespace(**hparams_dict["tuning"])

            if tuning_config.tune:
                if tuning_config.param_type == "boolean":
                    tunable_hparams[hparam] = hp.Boolean(name=hparam)
                elif tuning_config.param_type == "choice":
                    required_fields = ["values"]
                    if all(field in hparams_dict["tuning"].keys() for field in required_fields):
                        tunable_hparams[hparam] = hp.Choice(name=hparam, values=tuning_config.values)
                    else:
                        raise ValueError(
                            "Required fields for {} with param type {} are {}".format(
                                hparam,
                                tuning_config.param_type,
                                required_fields
                            )
                        )
                elif tuning_config.param_type == "float":
                    required_fields = ["min", "max", "step", "sampling"]
                    if all(field in hparams_dict["tuning"].keys() for field in required_fields):
                        tunable_hparams[hparam] = hp.Float(
                            name=hparam,
                            min_value=tuning_config.min,
                            max_value=tuning_config.max,
                            step=tuning_config.step,
                            sampling=tuning_config.sampling
                        )
                    else:
                        raise ValueError(
                            "Required fields for {} with param type {} are {}".format(
                                hparam,
                                tuning_config.param_type,
                                required_fields
                            )
                        )
                elif tuning_config.param_type == "int":
                    required_fields = ["min", "max", "step", "sampling"]
                    if all(field in hparams_dict["tuning"].keys() for field in required_fields):
                        tunable_hparams[hparam] = hp.Int(
                            name=hparam,
                            min_value=tuning_config.min,
                            max_value=tuning_config.max,
                            step=tuning_config.step,
                            sampling=tuning_config.sampling
                        )
                    else:
                        raise ValueError(
                            "Required fields for {} with param type {} are {}".format(
                                hparam,
                                tuning_config.param_type,
                                required_fields
                            )
                        )
                else:
                    raise ValueError("Invalid type for {}, must be boolean, choice, float or int.".format(hyperparam))
            else:  # i.e. T.tune==False
                tunable_hparams[hparam] = hp.Fixed(
                    name=hparam,
                    value=hparams_dict['default_value']
                )
        return tunable_hparams

    def build_model(self, tuner_hp) -> tf.keras.Model:
        """ Builds the model with the hyperparameters from the tuner.

        Args:
             tuner_hp (kt.HyperParameters): Keras Tuner hyperparameter object.

        Returns:
            model (keras.Model): Keras model object.
        """
        # Get set of tunable hyperparameters
        hparams = self.get_tunable_hparams(tuner_hp)
        # Build model
        model = generate_model(self.config, hparams)
        return model

    def build_tuner(self) -> kt.Tuner:
        """ Returns the Keras Tuner object.

        Returns:
            tuner (kt.Tuner): Keras Tuner object.

        """
        tuner = kt.RandomSearch(
            hypermodel=self.build_model,
            objective=self.config["params"]["objective"],
            max_trials=self.config["tuning"]["max_trials"],
            overwrite=True,
            directory="tuner_logs"
        )
        return tuner
