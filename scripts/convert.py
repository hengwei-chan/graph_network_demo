
import os
import shutil

def get_model_names():
    name_list = list()
    # two types: 1. all_logs_dgin3_1 2. only_logs_dgin3_1

    training_types = ['all_','only_'] # if all_logs, then no data_type needed
    data_types = ['logs_','logd_','logp_','logdp_','logds_','logsp_']
    # data_types = ['logp_','logds_']
    model_types = ['dgin','gin','dmpnn']
    feature_types = ['3_','4_','5_','6_','7_','8_']
    depths = ['2_','6_','8_']

    for training_type in training_types:
        if training_type == 'only_':
            for data_type in data_types:
                for model_type in model_types:
                    for feature_type in feature_types:
                        # for depth in depths:
                        name_list.append(training_type+data_type+model_type+feature_type+'1')
                        name_list.append(training_type+data_type+model_type+feature_type+'2')
        else:
            data_type = 'logs_'
            for model_type in model_types:
                for feature_type in feature_types:
                    # for depth in depths:
                    name_list.append(training_type+data_type+model_type+feature_type+'1')
                    name_list.append(training_type+data_type+model_type+feature_type+'2')
    return name_list

def getConfig(old_path,new_path):
    with open(old_path,'r') as old, open(new_path, "w") as new:
        for line in old:
            if ' test_model_weights_dir: str = ' in line:
                line = (line + '\n'+
                '    # To save the prediction values for each property set to True \n'
                '    # When this flag is True - the whole test dataset is taken an test_n_times is set to zero! \n'
                '    save_predictions: bool = False \n'
                '    # define the folder where you want to save the predictions. \n'
                '    # For each property, a file is created under the property name ("./logd.txt","./logs.txt","./logp.txt","./others.txt") \n'
                '    test_prediction_output_folder: str = project_path+"reports/predictions/"+model_name+"/" \n'
                )
            if ' include_logP: bool =' in line:
                line = (line+ '\n'+
                    '    include_other: bool = False \n'
                        )
            if 'from graphnets.utilities_chem import' in line:
                line = ('from graph_networks.utilities import * \n'
                )
            if '  project_path:str = ' in line:
                line = (
                    '    # path to the project folder \n'
                    '    project_path:str = "./" \n'
                )
            if '  logS_loss_mse = tf.keras.losses.mse' in line:
                line = (line+
                '    other_loss_mse = tf.keras.losses.mse \n'
                    )
            if '# project_path:str = ' in line:
                line = ('')
            if '  log_dir: str = project_path+' in line:
                line = (
                    '    log_dir: str = project_path+\'reports/logs/\'+model_name+\'.log\' \n'
                )
            if ' test_model_epoch: str =' in line:
                line= (line+'\n'+
                    '    # define the number or test runs for the CI. \n'+
                    '    # the mean and std of the RMSE and r^2 of the combined runs are taken as the output. \n'+
                    '    test_n_times: int = 1 \n'+
                    '    # do you want to test the model with consensus mode? \n'
                    '    # if yes, a defined ML model will be included in the consensus predictions during the testing. \n'
                    '    consensus: bool = False \n'
                     '    # include dropout during testing?\n'+
                    '    include_dropout: bool = False\n')
            # if ' consensus: bool =' in line:
            #     line= (line+'\n'+
            #     '    # include dropout during testing?\n'+
            #     '    include_dropout: bool = False\n')
            if ' best_evaluation_threshold: float' in line:
                line = (
                    '    # define the starting threshold for the RMSE of the model. When the comnbined RMSE \n'+
                    '    # is below this threshold, the model weights are being safed and a new threshold \n'+
                    '    # is set. It only serves as a starting threshold so that not too many models \n'+
                    '    # are being safed. Depends on how many log endpoints are being taken into \n'+
                    '    # consideration - as three endpoints have a higher combined RMSE as only one \n'+
                    '    # endpoint. \n'+
                    line+'\n'+
                    '    # define the individual thresholds. If one model is better, the corresponding \n'+
                    '    # model weights are being saved. \n'+
                    '    best_evaluation_threshold_logd: float = 1.85 \n'+
                    '    best_evaluation_threshold_logp: float = 1.65 \n'+
                    '    best_evaluation_threshold_logs: float = 2.15 \n' +
                    '    best_evaluation_threshold_other: float = 2.15 \n'
                    )
            if ' reduce_mean: bool =' in line:
                line = (line+'\n'+
                    '@dataclass \n'+
                    'class MLConfig: \n'
                    '    """ \n'+
                    '    Configs for the ML algorithm \n'+
                    '    """ \n'+
                    '    # which algorithm do you want to use for the consensus? \n'+
                    '    # possibilities are: "SVM", "RF", "KNN" or "LR" - all are regression models! \n'+
                    '        # SVM: Support Vector Machine; RF: Random Forest, KNN: K-Nearest Neigbors; LR: Linear Regression;\n'+
                    '    algorithm: str = "SVM" \n'+
                    '    # which fingerprint to use - possibilities are: "ECFP" or "MACCS" \n'+
                    '    fp_types: str = "ECFP" \n'+
                    "    # If 'ECFP' fingerprint is used, define the number of bits - maximum is 2048! \n"+
                    '    n_bits: int = 2048 \n'+
                    '    # If "ECFP" fingerprint is used, define the radius \n'+
                    '    radius: int = 4 \n'+
                    '    # define if descriptors should be included into the non-GNN molecular representation \n'+
                    '    include_descriptors: bool = True \n'+
                    '    # define if the descriptors should be standardizedby scaling and centering (Sklearn) \n'+
                    '    standardize: bool = True \n'
                        )
            if ' frag_acc_config: FrACConfig' in line:
                line= (line+'\n'+
                    '    ml_config: MLConfig \n')
            if ' safe_after_batch: int = ' in line:
                line = ('    # define the number of epochs for each test run.  \n'+
                        '    save_after_epoch: int = 3 \n'+
                        '    # dropout rate for the general model - mainly the MLP for the different log predictions \n'+
                        '    dropout_rate: float = 0.15 # the overall dropout rate of the readout functions \n'+
                        '    # the seed to shuffle the training/validation dataset; For the same dataset, even when \n'+
                        '    # combined_dataset is True, it is the same training/valiation instances \n'+
                        '    train_data_seed: int = 0 \n'
                        )
            new.write(line)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    names = get_model_names()
    # names = ['all_logs_dgin3_1','all_logs_dgin3_2']
    print("names",names)
    # old_path_config = '/home/owieder/Projects/graphnets/reports/configs/'+str(name)+'/' 
    # old_path_weights = '/home/owieder/Projects/graphnets/reports/model_weights/'+str(name)+'/' 
    
    # new_path_config = '/home/owieder/projects/graph_networks/reports/configs/'+str(name)+'/'
    # new_path_weights = '/home/owieder/projects/graph_networks/reports/model_weights/'+str(name)+'/'
    # if not os.path.exists(new_path_config):
    #     os.mkdir(new_path_config)
    # getConfig(old_path_config+str(name),
    #         new_path_config+str(name))
    # shutil.copyfile(new_path_config+str(name), new_path_config+'other_config.py')
    # # files = os.listdir(old_path_weights)
    # shutil.copytree(old_path_weights, new_path_weights)
    # shutil.copyfile(old_path_weights, new_path_weights)
    for name in names:
        try:
            old_path_config = '/home/owieder/Projects/graphnets/reports/configs/'+str(name)+'/' 
            # old_path_weights = '/home/owieder/Projects/graphnets/reports/model_weights/'+str(name)+'/' 
            
            new_path_config = '/home/owieder/projects/graph_networks/reports/configs/'+str(name)+'/'
            # new_path_weights = '/home/owieder/projects/graph_networks/reports/model_weights/'+str(name)+'/'
            if not os.path.exists(new_path_config):
                os.mkdir(new_path_config)
            getConfig(old_path_config+str(name),
                    new_path_config+str(name))
            shutil.copyfile(new_path_config+str(name), new_path_config+'other_config.py')
            # files = os.listdir(old_path_weights)
            # shutil.copytree(old_path_weights, new_path_weights)
        except Exception as e:
            print("could not convert due to: ",e)
            continue
    #     print("name",name)
    #     old_path = '/home/owieder/Projects/graphnets/reports/configs/'+str(name)+'/'+str(name) 
    #     new_path = '/home/owieder/projects/graph_networks/reports/configs/'+str(name)+'/'+str(name)
        # getConfig(old_path,
        #         new_path)
        # shutil.copyfile(new_path, PROJECT_PATH+'reports/configs/'+name+'/other_config.py')
