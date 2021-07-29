graph-networks
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master)


prediction of chemical properties with graphs and neural networks


### Installation
1. Clone the repository and `cd` into repository root:

    git clone https://github.com/spudlig/graph_networks.git
    cd graph_networks

2. Create a `conda` environment with the required dependencies:

    conda env create -f tf-cpu.yml

and activate it:

    conda activate tf-cpu


### Example usage
# Graph Generation
Only `.xls` files are currently able to produce graph instances.

To generate new graph instances for the Delaney dataset run the following line:

    python ./scripts/generate_graphs.py --input_file_path ./data/logd.xls --output_path_train ./data/pickled


# Generate graph instances
Use an xls file as an input - required arguments are:
1. --input_file_path    -set the input files
2. --output_path    -set the output folder (the graph instances are being pickled)
3. --featurization  -set what kind of featurization is needed
4. --columns    -select the columns to be read in

~ python ./graph_networks/scripts/generate_graphs.py --input_file_path ./data/lipo_plus.xls --output_path ./graph_networks/data/output_folder/ --columns 0 1 2 3 4
--featurization DGIN3

### Configuration
# Configuration file
The configuration file is composed of mainly 5 different sections and one general field.
    
    basic_model_config: BasicModelConfig
    model1_config: Model1Config
    d_gin_config: DGINConfig
    frag_acc_config: FrACConfig
    ml_config: MLConfig
    model: str = 'model10'

The `BasicModelConfig` dataclass defines the configurations for all graph neural network files and some general model parameters.
It includes the following parameters:

name of the model - unique identifier that needs to be different for each model as it
is the name under which everything is safed.

    model_name: str = 'only_logdp_dgin6_2'

batch size

    batch_size: int =15

if the model_name exists - should it override the existing model and all other things related to this model?

    override_if_exists: bool = True

path to the project folder

    project_path:str = './'

do you want to retrain an existing model_name model?

    retrain_model: bool = False

if True, define the name and epoch of the model

    retrain_model_name: str = ''
    retrain_model_epoch: str = ''

do not change this - is done automatically

    retrain_model_weights_dir: str = project_path+'reports/model_weights/'+retrain_model_name+'/epoch_'+retrain_model_epoch+'/checkp_'+retrain_model_epoch

define where the train and test data (pickled graphs) are located (folder only)
    train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logdp/'
    test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logdp/'

are there two different directories (e.g. when using two datasets that have not benn merged)
    combined_dataset: bool = False
if there are two different directories - define the second directory for train/test (folders only)
    add_train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logs/'
    add_test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logs/'

do you want to test the model?
    test_model: bool = False
if yes, define the epoch you want to use.
    test_model_epoch: str = '609'
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

the directory to the log files.
    log_dir: str = project_path+'reports/logs/'+model_name+'.log'
the verbosity of the logging
    verbosity_level = logging.INFO

what kind of model do you want to train - can be either "GIN", "MPNN" or "DGIN"
    model_type: str = "DGIN" 
    
do not change this! these paths are generated for the model under model_name
    plot_dir: str = project_path+'reports/figures/'+model_name+'/'
    tensorboard_log_dir: str = project_path+'reports/tensorboard/'+model_name+'/'
    config_log_dir: str = project_path+'reports/configs/'+model_name+'/'
    model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/'
    stats_log_dir: str = project_path+'reports/stats/'+model_name+'/'




# Train GNN with graph instances

### Copyright

Copyright (c) 2021, oliver wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.