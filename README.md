graph-networks
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master)


prediction of chemical properties with graphs and neural networks


# Installation
1. Clone the repository and `cd` into repository root:

    git clone https://github.com/spudlig/graph_networks.git
    cd graph_networks

2. Create a `conda` environment with the required dependencies:

    conda env create -f tf-cpu.yml

and activate it:

    conda activate tf-cpu


# Example usage
## Graph Generation
Only `.xls` files are currently able to produce graph instances.

To generate new graph instances for the Delaney dataset run the following line:

    python ./scripts/generate_graphs.py --input_file_path ./data/logd.xls --output_path_train ./data/pickled


## Generate graph instances
Use an xls file as an input - required arguments are:
1. --input_file_path    -set the input files
2. --output_path    -set the output folder (the graph instances are being pickled)
3. --featurization  -set what kind of featurization is needed
4. --columns    -select the columns to be read in

~ python ./graph_networks/scripts/generate_graphs.py --input_file_path ./data/lipo_plus.xls --output_path ./graph_networks/data/output_folder/ --columns 0 1 2 3 4
--featurization DGIN3

# Configuration
## Configuration file
The configuration file is composed of 5 different sub configuration parts and one general field.
The sub configurations are:
    
    basic_model_config: BasicModelConfig
    model1_config: Model1Config
    d_gin_config: DGINConfig
    frag_acc_config: FrACConfig
    ml_config: MLConfig

The model type should be set to model10 to recreate the publication results

    model: str = 'model10'

A detailed description of the other sub configurations can be found here:

### Basic Model Config
The `BasicModelConfig` dataclass defines the configurations for all graph neural network files and some general model parameters.
It includes the following parameters:

name of the model - unique identifier that needs to be different for each model as it
is the name under which everything is safed.

    model_name: str = 'only_logdp_dgin6_2'

batch size

    batch_size: int =15

flag to override the existing model.

    override_if_exists: bool = True

the absolut project path

    project_path:str = './'

flag to retrain the model

    retrain_model: bool = False

if True, define the name and epoch of the model

    retrain_model_name: str = ''
    retrain_model_epoch: str = ''

model weights when retraining is done. Set automatically

    retrain_model_weights_dir: str = project_path+'reports/model_weights/'+retrain_model_name+'/epoch_'+retrain_model_epoch+'/checkp_'+retrain_model_epoch

define where the train and test data (pickled graphs) are located (folder only)

    train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logdp/'
    test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logdp/'

are there two different directories (e.g. when using two datasets that have not benn merged)

    combined_dataset: bool = False

if there are two different directories - define the second directory for train/test (folders only)

    add_train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logs/'
    add_test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logs/'

flag to test the model

    test_model: bool = False

if test is True, define the epoch you want to use for testing

    test_model_epoch: str = ''

define the number or test runs for the CI.
the mean and std of the RMSE and r^2 of the combined runs are taken as the output.
    
    test_n_times: int = 1

do you want to test the model with consensus mode?
if yes, a defined ML model will be included in the consensus predictions during the testing.

    consensus: bool = False

include dropout during testing?

    include_dropout: bool = False

do not change this! it will be automatically updated to use the epoch and weights of the model.
if no weights for the epoch are present, there will be an error.

    test_model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/epoch_'+test_model_epoch+'/checkp_'+test_model_epoch

To save the prediction values for each property set to True
When this flag is True - the whole test dataset is taken an test_n_times is set to zero!
    
    save_predictions: bool = False

define the folder where you want to save the predictions.
For each property, a file is created under the property name ('./logd.txt','./logs.txt','./logp.txt','./others.txt')

    test_prediction_output_folder: str = project_path+'reports/predictions/'+model_name+'/'

the directory to the log files

    log_dir: str = project_path+'reports/logs/'+model_name+'.log'

the verbosity of the logging. Can be set to `logging.DEBUG`,`logging.INFO`,`logging.WARNING`,`logging.ERROR`,`logging.CRITICAL`

    verbosity_level = logging.INFO

what kind of model do you want to train - can be either `GIN`, `MPNN` or `DGIN`

    model_type: str = "DGIN" 
    
the created paths for each model under which one can find plots, tensorboard logs, configuration files, model weights and statistics.

    plot_dir: str = project_path+'reports/figures/'+model_name+'/'
    tensorboard_log_dir: str = project_path+'reports/tensorboard/'+model_name+'/'
    config_log_dir: str = project_path+'reports/configs/'+model_name+'/'
    model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/'
    stats_log_dir: str = project_path+'reports/stats/'+model_name+'/'


### Model1 Config
The `Model1Config` dataclass defines the general configurations for the GNN.

how the training and validation should be split; e.g. 0.90 are 90% training, 10% validation

    validation_split: float = 0.90

specific learning rate and clip rate

    learning_rate: float = 0.004
    clip_rate: float = 0.6

optimizer

    optimizer = tf.keras.optimizers.Adam(learning_rate)

different losses - only MSE

    lipo_loss_mse = tf.keras.losses.mse
    logP_loss_mse = tf.keras.losses.mse
    logS_loss_mse = tf.keras.losses.mse
    other = tf.keras.losses.mse
    
define how many epochs should be run

    epochs: int = 1600

define the number of epochs for each test run. 

    save_after_epoch: int = 3

dropout rate for the general model - mainly the MLP for the different log predictions

    dropout_rate: float = 0.15 # the overall dropout rate of the readout functions

the seed to shuffle the training/validation dataset; For the same dataset, even when
combined_dataset is True, it is the same training/valiation instances
    
    train_data_seed: int = 0

hidden feature output size of the first layer in the MLP for the different log predictions - can be changed

    hidden_readout_1: int = 32

hidden feature output size of the second layer in the MLP for the different log predictions - can be changed

    hidden_readout_2: int = 14

activation function for the MLP for the different log predictions.
can be changed

    activation_func_readout = tf.nn.relu
    
define what property prediction should be included.
If set True but does not have that property, an exception is thrown BUT the model
continuous 

    include_logD: bool = True
    include_logS: bool = False
    include_logP: bool = True
    include_other: bool = True

define the starting threshold for the RMSE of the model. When the comnbined RMSE
is below this threshold, the model weights are being safed and a new threshold
is set. It only serves as a starting threshold so that not too many models
are being safed. Depends on how many log endpoints are being taken into
consideration - as three endpoints have a higher combined RMSE as only one 
endpoint.

    best_evaluation_threshold: float = 2.55

define the individual thresholds. If one model is better, the corresponding
model weights are being saved.

    best_evaluation_threshold_logd: float = 1.85
    best_evaluation_threshold_logp: float = 1.65
    best_evaluation_threshold_logs: float = 2.15
    best_evaluation_threshold_other: float = 2.15

do you want to use RMSE instead of MSE for the different losses?
True if yes, False if not

    use_rmse: bool = True

reshuffles the training data set in each epoch

    shuffle_inside: bool = True

### D-GIN Config
The `DGINConfig` dataclass defines the configurations for the specific submodel configs of the DGIN, MPNN or GIN model.

initialize the DGIN or D-MPNN features with bias; the GIN's features are initialized without NN

    init_bias: bool = False
    
DMPNN part:
include during aggregate part

    dropout_aggregate_dmpnn: bool = False
    layernorm_aggregate_dmpnn: bool = True
    dmpnn_passing_bias: bool = False

include during message passing part

    dropout_passing_dmpnn: bool = False
    layernorm_passing_dmpnn: bool = True

how many layers during the D-MPNN message iteration phase in the D-GIN or only D-MPNN model

    massge_iteration_dmpnn: int = 4

GIN part:
include during aggregate part

    dropout_aggregate_gin: bool = False
    layernorm_aggregate_gin: bool = True
    gin_aggregate_bias: bool = False

include during passing part

    dropout_passing_gin: bool = False
    layernorm_passing_gin: bool = True

how many layers during the GIN message iteration phase in the D-GIN or only GIN model

    message_iterations_gin: int = 4

dropout used throughout the models

    dropout_rate: float = 0.15

input dimension of the models; is generated automatically so DO NOT CHANGE

    input_size: int = (ATOM_FEATURE_DIM+EDGE_FEATURE_DIM) # combination of node feature len (33) and edge feature len (12)

can be changed - hidden feature size

    passing_hidden_size: int = 56

dimension of GIN input in the D-GIN or at the end of the GIN model during the aggregation phase; is generated automatically so DO NOT CHANGE

    input_size_gin: int = (ATOM_FEATURE_DIM+passing_hidden_size)

### Machine Learning Configs
The `MLConfig` dataclass defines the configurations for the ML algorithms during the consensus scoring.

which algorithm do you want to use for the consensus?
possibilities are: 'SVM', 'RF', 'KNN' or 'LR' - all are regression models!
SVM: Support Vector Machine; RF: Random Forest, KNN: K-Nearest Neigbors; LR: Linear 
Regression;

    algorithm: str = 'SVM'

which fingerprint to use - possibilities are: 'ECFP' or 'MACCS'

    fp_types: str = 'ECFP'

If 'ECFP' fingerprint is used, define the number of bits - maximum is 2048!

    n_bits: int = 2048

If 'ECFP' fingerprint is used, define the radius

    radius: int = 4

define if descriptors should be included into the non-GNN molecular representation

    include_descriptors: bool = True

define if the descriptors should be standardizedby scaling and centering (Sklearn)

    standardize: bool = True


### Fragmentation Config
The `FrACConfig` dataclass is not used for the Model1.


### Copyright

Copyright (c) 2021, oliver wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.