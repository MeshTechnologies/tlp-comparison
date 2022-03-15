# Cookiecutter for ML Projects with DVC + CML

## Table of Contents
1. [Repo Structure](#structure)
2. [Steps to Set Up a New Repo](#steps)
   1. [Import and Track Input Artifacts with DVC](#import)
   2. [Prepare Data](#data)
   3. [Create Model Objects](#models)
   4. [Compile Models for Training with ModelTrainer](#trainer)
   5. [Create the Script for Running a Train/Tune Job](#run)
   6. [Configuring Your Data, Models, and Runs](#config)
   7. [Specify Requirements](#reqs)
   8. [List Inputs and Outputs for DVC](#deps)
   9. [Running Locally](#local)
   10. [Running on a VM](#vm)

## Repo Structure <a id="structure"></a>
```
.
├── README.md
├── config.yaml
├── data
│   ├── processed
│   └── raw
├── models
│   └── [YOUR MODEL NAME]
│       ├── config.yaml
│       ├── dvc.yaml
│       ├── hparam_config_example.yaml
│       └── hyperparameters.yaml
├── requirements.txt
├── run.sh
├── src
│   ├── data
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── data_provider.py
│   ├── models
│   │   ├── README.md
│   │   └── __init__.py
│   ├── notebooks
│   ├── run.py
│   ├── trainer.py
│   ├── tuner.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── training_metrics
└── training_metrics.json
```

## Steps to Set Up a New Repo <a id="steps"></a>
### 1: Import and Track Input Artifacts with DVC <a id="import"></a>
Input artifacts are files/dirs that are required as inputs to your training pipeline. These will need to be manually 
downloaded/generated locally for debugging but to run our training on a VM we will need to include downloading logic in
our CI script. This is covered in Step 9. As for the artifacts themselves, they come in two types:

**External Artifacts** <br/>
External artifacts are any files/dirs that were generated and DVC-tracked in another repo. Usually this will only be 
datasets from our [Dataset Registry](https://github.com/MeshTechnologies/dataset-registry), but could also be objects 
such as tokenizers or pre-trained models. 

To import these objects navigate to the location you'd like to put the objects and then use: <br /> 
`dvc import [REPOSITORY GIT URL] [PATH TO FILE/DIR]`

For example: <br/>
```shell
cd data/processed
dvc import  git@github.com:MeshTechnologies/dataset-registry.git processed/code/01-glt_30_languages/30langs_200lines_fragmented
```

[dvc import](https://dvc.org/doc/command-reference/import) downloads the raw object from the other repo as well as 
DVC-tracks it. This will generate a `.dvc` file and add the raw file/dir to a `.gitignore` so we can version control the 
dependency without storing the large object itself.

**Internal Artifacts** <br/>
These are files/dirs that are local to the current repo, such as tokenizers. These objects are too large to be tracked
with Git, so we will track them with DVC and then `dvc pull` them when training on the VM (see Step 9). See the DVC docs
on [adding](https://dvc.org/doc/command-reference/add) a DVC file and [pushing](https://dvc.org/doc/command-reference/push)
that file to our remote GCS bucket.

### 2: Prepare Data <a id="data"></a>
All data loading and processing should occur in `src/data`. Obviously this will change project-to-project, but the only 
requirement is that the final dataset(s) be added as `DataProvider` [attributes](https://realpython.com/lessons/class-and-instance-attributes/) 
via `get_data()` in `src/data/data_provider.py`.

Feel free to create as many helper classes, scripts, or functions in `src/data` and/or `data_provider.py`. The final 
datasets will utilized in `src/run.py`.

### 3: Create Model Objects <a id="models"></a>
Models should be created in `src/models` where they will eventually be imported and used in `src/trainer.py`. Per the 
example in the `src/models` [README](src/models/README.md), I found it easiest to create simple functions that return 
`Keras.Model`objects, but feel free to utilize classes as well. 

When creating your models, try not to hardcode training parameters (epochs, batch size, etc.) or model hyperparameters
(dropout rate, hidden units, etc.). These (hyper)parameters should be specified in the `config.yaml` so we can trak the
parameter sets as artifacts for experiments and to enable hyperparameter tuning. See Step 6 for more information. 

### 4: Compile Models for Training with ModelTrainer <a id="trainer"></a>
All logic for compiling models, training, evaluating, and saving should be done in `src/trainer.py`. 

All methods here are pretty standard (add more as needed) besides `generated_model()`. This method is called in the 
constructor of a `ModelTrainer` instance and assigns the `model` attribute. This is where you will call your 
functions/classes from `src/models`, using the `config.yaml` (hyper)params to construct the desired model itself as well
as other necessary fields for compiling a Keras model, such as loss, optimizer, and metrics.

The `ModelTrainer`class will be imported in `src/run.py`.

### 5: Create the Script for Running a Train/Tune Job <a id="run"></a>
`src/run.py` is the script that will be called for tuning and/or training. It is here where the `DataProvider` (Step 2)
and the `ModelTrainer` (Step 4) classes will be instantiated.

The default file should have most, if not all, of the code you'll need for training/tuning. We will cover configuring 
hyperparameters for tuning in Step [6].

### 6: Configuring Your Data, Models, and Runs <a id="config"></a>
Our data processing, models, and training runs all require configurable parameters. Rather than hardcoding these values
in the Python files, we will define them in a `config.yaml`. This allows us to create a config artifact that we attach 
to a specific run so we can maintain reproducibility and more easily compare runs.

The default config is located at `models/[YOUR MODEL NAME]/config.yaml`. Before editing this file, first rename the
`[YOUR MODEL NAME]` to a generic name for the model you're working on (e.g. language_classifier, is_code, color_extraction).

Ignore the `inputs_paths` and `output_paths` sections. We will cover these in Step [X]. 

**params** <br/>
The `params` section is where you will put all global, non-tunable parameters related to data or training.

**tuning** <br/>
The `tuning` section is where you will configure all model hyperparamters as well as tuning parameters.

Set `tune` to `true` if you'd like to run a hyperparameter tuning sweep and set the maximum number of trials you'd like
to run for that sweep.

**tuning:hyperparams** <br/>
This is where you will specify the hyperparameters for you model(s). If you know you won't need to tune a parameter you
will only need to specify a `default_value` attribute. For example:
```yaml
learning_rate:
      default_value: 0.0001
```
If you want to tune the hyperparameter you'll still need a `default_value`, but you'll also have to create a `tuning` 
subsection. We use the KerasTuner for tuning, so these subsections will have all required parameters for a 
[kt.Hyperparameters object](https://keras.io/api/keras_tuner/hyperparameters/). You'll need to specify whether a 
parameter is an int, float, boolean or choice. The tuner will require a different set of input parameters for each type. 
Copy the formatting from a similar hyperparameter in examples in the `models/[YOUR MODEL NAME]` [README](src/models/README.md) 
to ensure compatibility. For example:
```yaml
embedding_dim:
  default_value: 10
  tuning:
    tune: False
    param_type: int
    min: 64
    max: 128
    step: 32
    sampling: linear # "linear", "log", or "reverse_log"
```

### 7: Specify Requirements <a id="require"></a>
List all required packages (versions optional) in `./requirements.txt`. These will be installed automatically on the VMs,
but you will be responsible for installing them locally for debugging.

### 8: List Inputs and Outputs for DVC <a id="deps"></a>
`models/[YOUR MODEL NAME]/dvc.yaml` is our pipeline configuration. This is where we specify the stages, dependencies, and
outputs for our pipeline. The only sections you'll likely need to change here are the input dependencies 
(`stages/:train:deps`) and the output artifacts ('stages:train:outs`). 

**Input Dependencies** <br/>
These are all Python scripts and objects, likely DVC-tracked, (models, tokenizers, etc.) required to run the `train` 
stage. Notice how we use an `input_paths` variable. We skipped this in Step 6, but you'll now need to add all required 
paths to `input_paths` in `models/[YOUR MODEL NAME]/config.yaml`.

**Output Artifacts** <br/>
Here is where we tell DVC what artifacts will be output by the `train` stage. These will be DVC-tracked and pushed to 
remote storage via CI. Like our Input Dependencies, you'll need to specify the paths to these artifacts in
`output_paths` in `models/[YOUR MODEL NAME]/config.yaml`. 

If you'd like to track an artifact with Git rather than DVC, use the `cache: false` option:
```yaml
outs:
  - ${output_paths.model.tflite_model}
  - ${output_paths.model.keras_model}
  - ${output_paths.model.output_mapping}
  - ${output_paths.model.root}/hyperparameters.yaml:
      cache: false
```

### 8: Running Locally <a id="local"></a>
Assuming all requirements are installed and input artifacts are downloaded, to run a train/tune job on your local
machine simple run `sh run.sh` from the root of your project.

### 9: Running on a VM <a id="vm"></a>
To train/tune on GCP VM we will use Iterative's [CML](https://cml.dev/) + 
[GitHub Actions](https://docs.github.com/en/actions). This pipeline is managed by `./.github/workflows/cml.yaml`. The 
majority of this file should be left untouched besides the following:

**Specifying Machine Type** <br/>
First, we must choose what type of GCP VM we'd like to use. Navigate to `jobs:deploy_runner:steps:run` to change these
settings in the `cml runner` command.

Choose your region from [this list](https://cloud.google.com/compute/docs/regions-zones). `cloud_type` is the type of 
[GCP machine](https://gcpinstances.doit-intl.com/) to use + the number of vCPU's. `cloud_gpu` is the 
[GCP GPU](https://cloud.google.com/compute/docs/gpus) accelerator. If you'd like to just train on a CPU, use `nogpu`
as the value.

**Reusing a VM** <br/>
It's useful to keep a VM spun up so that you can keep packages and DVC files downloaded there. To do this, add the 
`--reuse` flag to `cml runner`. What this does is look for a machine with the label(s) from `cml runner`. It's 
recommended to set your own machine label because if you use `--reuse` it will use a VM even if it's in the middle of 
another training job. If you set your own label, you must tell GitHub Actions what machine to look for. Add that label 
`jobs:train:runs-on`.

**Download DVC-tracked Input Artifacts** <br/>
The first part for this was covered in Step 1 where the DVC file(s) are downloaded and tracked with `dvc import` or 
manually added to DVC from the current repo using `dvc add` and `dvc push`. This is sufficient for local debugging, but 
we'll need to download these same artifacts on the VM. 

Navigate to `jobs:train:steps` and then the `Get Input Artifacts` step to add these downloads. 

For the external artifacts, we'll use `dvc update [PATH TO ARTIFACT].dvc`. `dvc update` is the pair command to
`dvc import` for downloading from external repos.

For the internal artifacts, assuming they've first been pushed to remote, we'll use `dvc pull [PATH TO ARTIFACT]`. This
simply downloads the artifact using the `.dvc` file at that path (but don't use the `.dvc` file in the command).

**Git-tracking Some Output Artifacts** <br/>
There are some cases where output artifacts are small enough to be git-tracked. For example, by default we git-track
the `training_metrics/` dir and the `training_metrics.json` summary. Other examples could be label mappings, plots, etc.

If you have files that you'd like to push to git following the completion of a training job, navigate to 
`jobs:train:steps` and then the `Train Model` step. Under "run", find the comment that says 
"# Push all updated files, skipping CI" and `git add ` your files here. 

**Starting a VM Train Job** <br/>
When you're ready to start a training job on a VM, commit and push your local changes with a commit message that 
includes `[cml]` at the start. GitHub Actions will look for this substring and start a job if it sees it. If it's not 
present in a commit message, then it will behave like a typical commit.
