import tensorflow as tf


class DataProvider:
    """ Base class for loading and preprocessing data.

    Attributes:
        config (dict): The configuration for the train/tune run from config.yaml
        params (dict): The non-tunable parameters from config.yaml:params

    """

    def __init__(self, config):
        self.config = config
        self.params = config["params"]
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.get_data()

    def get_data(self):
        """ Loads and prepare the dataset(s) for training.
        The final datasets should be assigned to DataProvider Attributes (e.g. self.train_ds = train.ds)
        """

        dataset_root = self.config['input_paths']['data']['root']



