# Example DataProvider Class
This class is used to provide data in `src/run.py`.

**`data_provider.py`**
```python
import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer
import os


class DataProvider:
    """ Base class for loading and processing data.

    Attributes:
        config (dict): The configuration for the train/tune run.
        params (dict): The non-tunable parameters
        labels (list): The labels for the data
        n_classes (int): The number of classes

    """

    def __init__(self, config):
        self.config = config
        self.params = config["params"]
        self.labels = self.get_labels()
        self.n_classes = len(self.labels)

        print("Loading raw data")
        # Load raw data
        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.config["input_paths"]["dataset"]["train"],
            batch_size=self.params["batch_size"],
            seed=self.params["seed"],
        )
        raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.config["input_paths"]["dataset"]["valid"],
            batch_size=self.params["batch_size"],
            seed=self.params["seed"],
        )
        raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.config["input_paths"]["dataset"]["test"],
            batch_size=self.params["batch_size"],
            seed=self.params["seed"],
        )

        print("Loading tokenizer")
        # Load tokenizer
        tokenizer_path = self.config["input_paths"]["tokenizer_json"]
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.vocab_size = len(list(self.tokenizer.get_vocab().keys()))
        self.max_seq_len = self.params["sequence_len"]
        self.tokenizer.enable_truncation(max_length=self.max_seq_len)
        self.tokenizer.enable_padding(length=self.max_seq_len)

        print("Precaching data")
        # Precache data
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = raw_train_ds.map(self.tf_tokenize, num_parallel_calls=AUTOTUNE)
        val_ds = raw_val_ds.map(self.tf_tokenize, num_parallel_calls=AUTOTUNE)
        test_ds = raw_test_ds.map(self.tf_tokenize, num_parallel_calls=AUTOTUNE)

        # self.train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # self.val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # self.test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        self.train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    def get_labels(self) -> list:
        """ Get the labels for the data.

        Returns:
            labels (list): The labels for the data

        """
        # Get class labels from directory names, mirroring functionality
        # of tf.keras.preprocessing.text_dataset_from_directory
        labels = os.listdir(self.config["input_paths"]["dataset"]["train"])
        labels = [label for label in labels if "." not in label]
        labels.sort()
        return labels

    def tokenize(self, batch) -> np.ndarray:
        """ Tokenize a batch of text.

        Args:
            batch (list): A batch of text

        Returns:
            np.ndarry: The tokenized batch

        """
        ids = []
        for example in batch:
            example_str = example.numpy().decode(encoding="utf-8", errors='ignore')
            example_ids = self.tokenizer.encode(example_str).ids
            ids.append(example_ids)
        return np.array(ids)

    def tf_tokenize(self, text, label):
        # text = tf.expand_dims(text, -1)
        result_text = tf.py_function(func=self.tokenize, inp=[text], Tout=tf.int64)
        # result_pt.set_shape([None])
        return result_text, label
```