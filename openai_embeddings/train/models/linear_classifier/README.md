# Example Hyperparameter Config

**`config.yaml:tuning`**
```yaml
tuning:
  tune: Tune
  max_trials: 2

  hyperparams:
    model_architecture:
      default_value: vanilla_LSTM
      tuning:
        param_type: choice
        tune: False
        values:
          - vanilla_LSTM
          - att_LSTM

    loss:
      default_value: Crossentropy
      tuning:
        param_type: choice
        tune: False
        values:
          - Crossentropy

    optimizer:
      default_value: Adam
      tuning:
        param_type: choice
        tune: False
        values:
          - Adam

    bidirectional: # boolean, whether LSTM is bidirectional or not
      default_value: False
      tuning:
        tune: False
        param_type: boolean

    embedding_dim:
      default_value: 10
      tuning:
        tune: False
        param_type: int
        min: 64
        max: 128
        step: 32
        sampling: linear # "linear", "log", or "reverse_log"

    hidden_dim:
      default_value: 10
      tuning:
        tune: False
        param_type: int
        min: 64
        max: 128
        step: 32
        sampling: linear

    att_units:
      default_value: 64
      tuning:
        tune: False
        param_type: int
        min: 64
        max: 128
        step: 32
        sampling: linear


    dropout_p:
      default_value: .05
      tuning:
        tune: True
        param_type: float
        min: 0.01
        max: 0.5
        step: #optional
        sampling: log

    learning_rate:
      default_value: 0.0001
      tuning:
        tune: False
        param_type: float
        min: 0.0001
        max: 0.1
        step:  #optional
        sampling: log
```