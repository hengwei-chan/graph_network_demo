import logging
log = logging.getLogger(__name__)

import numpy as np
import os
import datetime
import time
import pandas as pd
import random
import sys
import copy

import tensorflow as tf
from tensorflow.keras.layers import Dense as dense
from tensorboard.plugins import projector
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from graph_networks.model import BaseModel
from graph_networks.utilities import write_out, fit_model, test_model, get_r2, make_batch


class Model10(BaseModel):
    """[summary]
    This model is used to train, validate and test logD logS and logP endpoints by deploying
    either D-MPNN, GIN or D-GIN architectures.
    It is an extention of the base model class.
    """
    def __init__(self,config,name="Model-10"):
        super().__init__(config)
        self.consensus = config.basic_model_config.consensus

        self.best_test_overall = self.config.best_evaluation_threshold
        self.best_test_logd = self.config.best_evaluation_threshold_logd
        self.best_test_logs = self.config.best_evaluation_threshold_logs
        self.best_test_logp = self.config.best_evaluation_threshold_logp


    def call(self, instance, train=True):
        '''
        each method (train/validate/test) calls this.
        INPUT:\n
            instance (AtomGraph): molecular graph \n
            train (bool): is it called during training or testing \n
        RETURNS:\n
            (dict) predictions for the instance (containing logP, logD and logS float values)\n
            (tensor list) encoding of the instance 
        '''

        predictions = {'logP_pred':0.0,'logD_pred':0.0,'logS_pred':0.0}

        dgin_encoding = self.dgin.call(instance,train=train)
        
        if self.config.include_logD:
            predictions['logD_pred'] = self.logD(dgin_encoding,training=train)
        if self.config.include_logP:
            predictions['logP_pred'] = self.logP(dgin_encoding,training=train)
        if self.config.include_logS:
            predictions['logS_pred'] = self.logS(dgin_encoding,training=train)
        return predictions,dgin_encoding

    def run(self,data,test_data):
        '''
        the center method of the model for training/evalaution.
        in the train/validate method, the call method is called.\n
        INPUT:\n
            data (list of AtomGraph): the dataset of batched AtomGraphs.
            Consist of the training and evaluation dataset \n
        RETURNS:\n
            None
        '''
        train_summary_writer = tf.summary.create_file_writer(self.tensorboard_log_dir)
        val_summary_writer = tf.summary.create_file_writer(self.tensorboard_log_dir+'val')
        train_d= data[0]
        eval_d = data[1]

        self.consensus = False
        logging.debug("Consensus should only be set to True when testing, not training - It was set to False!")
            
        overall_batch_iterations = 0
        for epoch in range(0,self.config.epochs):
            if self.config.shuffle_inside:
                random.shuffle(train_d)
            start_time = time.time()
            nr_batches = 0
            
            for batch in train_d[:100]:
                try:
                    self.train(batch,overall_batch_iterations,train_summary_writer)
                    nr_batches += 1
                    overall_batch_iterations += 1
                    sys.stdout.write("\r{0}".format((float(nr_batches)/len(train_d))*100))
                    sys.stdout.flush() # not on cluster
                except Exception as e:
                    logging.error('train error during batch nr.'+str(nr_batches)+' in epoch: '+str(epoch))
                    print("train error",e,'during batch nr.',nr_batches,'in epoch:',epoch)
                    continue
            #### EVALUATE - after each epoch
            self.evaluate(eval_d,epoch,val_summary_writer,test_data)
            elapsed_time_fl = (time.time() - start_time) 
            print("\n!!!!!!!!!!!!!!!!!!!!!!!EPOCH!!!",self.model_name,'\n',epoch,elapsed_time_fl)
            print("best EVALUTION (overall, logd, logs, logp): "+str(self.best_validation_overall)+'_'+str(self.best_validation_logd)+'_'+
            str(self.best_validation_logs)+'_'+str(self.best_validation_logp))
            logging.debug("best EVALUTION:",self.best_validation_overall)
            logging.debug("\n EPOCH: "+str(epoch)+' for model ' +str(self.model_name)+' within '+str(elapsed_time_fl))
            
            if epoch % self.config.safe_after_batch == 0 and epoch > 1:
                self.test(test_data,epoch=epoch)
        return

    def train(self, batch,overall_batch_iterations,train_summary_writer):
        '''
        Trains the model and updates the weights after each batch with the defined optimizer. \n
        The results are only saved as tensorboard files not text files. \n
        INPUT:\n
            batch (list of AtomGraphs): training batch of molecular graphs \n
            overall_batch_iterations (int): overall number of batches - for tensorboard \n
            train_summary_writer (tf summary file): to save the loss \n
        RETURNS:\n
            None \n 
        '''
        batch_loss = list()
        with tf.GradientTape() as tape:
            gather_logD_pred,gather_logD_exp = list(),list()
            gather_logP_pred,gather_logP_exp = list(),list()
            gather_logS_pred,gather_logS_exp = list(),list()
            for instance in batch:
                predictions, dgin_encoding = self.call(instance,train=True)
                ## logD prediction
                if self.config.include_logD:
                    if instance.properties['logd']:
                        gather_logD_exp.append(tf.convert_to_tensor(instance.properties['logd']))
                        gather_logD_pred.append(predictions['logD_pred'][0][0])
                ## logS prediction
                if self.config.include_logS:
                    if instance.properties['logs']:
                        gather_logS_exp.append(tf.convert_to_tensor(instance.properties['logs']))
                        gather_logS_pred.append(predictions['logS_pred'][0][0])

                ## logP prediction
                if self.config.include_logP:
                    if instance.properties['logp']:
                        gather_logP_exp.append(tf.convert_to_tensor(instance.properties['logp']))
                        gather_logP_pred.append(predictions['logP_pred'][0][0])
            # logD
            logD_loss = tf.convert_to_tensor(0.0)
            if self.config.include_logD:
                if instance.properties['logd']:
                    logD_loss = tf.math.sqrt(self.lipo_loss_mse(gather_logD_exp,gather_logD_pred))
            # logP
            logP_loss = tf.convert_to_tensor(0.0)
            if self.config.include_logP:
                logP_loss = tf.math.sqrt(self.logP_loss_mse(gather_logP_exp,gather_logP_pred))
            # logS
            logS_loss = tf.convert_to_tensor(0.0)
            if self.config.include_logS:
                if instance.properties['logs']:
                    logS_loss = tf.math.sqrt(self.logS_loss_mse(gather_logS_exp,gather_logS_pred))
            ### overall
            batch_loss = logD_loss+logP_loss+logS_loss
            grads = tape.gradient(tf.reduce_mean(batch_loss), self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip_rate)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # save the loss after 50 batches
        if overall_batch_iterations % 50 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('train overall loss', batch_loss.numpy(), step=overall_batch_iterations)
                if self.config.include_logD:
                    if logD_loss.numpy() > 0.0:
                        tf.summary.scalar('train logD loss', logD_loss.numpy(), step=overall_batch_iterations)
                if self.config.include_logP:
                    tf.summary.scalar('train logP loss', logP_loss.numpy(), step=overall_batch_iterations)
                if self.config.include_logS:
                    if logS_loss.numpy() > 0.0:
                        tf.summary.scalar('train logS loss', logS_loss.numpy(), step=overall_batch_iterations)

    def evaluate(self, evaluation_data,epoch,val_summary_writer,test_data):
        '''
        evaluates the trained model. The results are being saved in tensorboard files
        as well as in text files.\n
        INPUT:\n
            evaluation_data (list of AtomGraphs): evaluation data consisting of AtomGraph instances \n
            epoch (int): current epoch - for tensorboard \n
        RETURNS:\n
            None \n 
        '''
        logging.debug("Evaluation starts!")
        gather_logD_pred,gather_logD_exp = list(),list()
        gather_logP_pred,gather_logP_exp = list(),list()
        gather_logS_pred,gather_logS_exp = list(),list()

        for batch in evaluation_data:
            for instance in batch:
                predictions,dgin_encoding = self.call(instance,train=False)
                ## logD prediction - for plot and current losses
                if self.config.include_logD:
                    if instance.properties['logd']:
                        gather_logD_exp.append(instance.properties['logd'])
                        gather_logD_pred.append(predictions['logD_pred'][0][0])
                ## logS prediction - for plot and current losses
                if self.config.include_logS:
                    if instance.properties['logs']:
                        gather_logS_exp.append(instance.properties['logs'])
                        gather_logS_pred.append(predictions['logS_pred'][0][0])

                ## logP prediction - for plot and current losses
                if self.config.include_logP:
                    if instance.properties['logp']:
                        gather_logP_exp.append(instance.properties['logp'])
                        gather_logP_pred.append(predictions['logP_pred'][0][0])
        # logD
        logD_loss = tf.convert_to_tensor(0.0)
        if self.config.include_logD:
            logD_loss = tf.math.sqrt(self.lipo_loss_mse(gather_logD_exp,gather_logD_pred))
        # logP
        logP_loss = tf.convert_to_tensor(0.0)
        if self.config.include_logP:
            logP_loss = tf.math.sqrt(self.logP_loss_mse(gather_logP_exp,gather_logP_pred))
        # logS
        logS_loss = tf.convert_to_tensor(0.0)
        if self.config.include_logS:
            logS_loss = tf.math.sqrt(self.logS_loss_mse(gather_logS_exp,gather_logS_pred))
        ### overall
        overall_evaluation_loss = logD_loss+logP_loss+logS_loss
        better = False
        if float(overall_evaluation_loss.numpy()) < float(self.best_validation_overall):
            print("BETTER!!!!!! best value VALIDATION and current loss (epoch)",self.best_validation_overall,overall_evaluation_loss,epoch)
            self.save_weights(self.model_weights_dir+'eval/'+'checkp_'+str(epoch))
            self.best_validation_overall = copy.copy(overall_evaluation_loss.numpy())
            better = True
        
        if float(logD_loss.numpy()) < float(self.best_validation_logd):
            print("BETTER!!!!!! best value LOGD VALIDATION and current loss (epoch)",logD_loss,epoch)
            self.save_weights(self.model_weights_dir+'eval/logd/'+'checkp_'+str(epoch))
            self.best_validation_logd = copy.copy(logD_loss.numpy())
            write_out(self.stats_log_dir+'eval/',self.model_name,('logD: '+str(self.best_validation_logd)),epoch)
            better = True
            
        if float(logS_loss.numpy()) < float(self.best_validation_logs):
            print("BETTER!!!!!! best value LOGS VALIDATION and current loss (epoch)",logS_loss,epoch)
            self.save_weights(self.model_weights_dir+'eval/logs/'+'checkp_'+str(epoch))
            self.best_validation_logs = copy.copy(logS_loss.numpy())
            write_out(self.stats_log_dir+'eval/',self.model_name,('logS: '+str(self.best_validation_logs)),epoch)
            better = True
        
        if float(logP_loss.numpy()) < float(self.best_validation_logp):
            print("BETTER!!!!!! best value LOGP VALIDATION and current loss (epoch)",logP_loss,epoch)
            self.save_weights(self.model_weights_dir+'eval/logp/'+'checkp_'+str(epoch))
            self.best_validation_logp = copy.copy(logP_loss.numpy())
            write_out(self.stats_log_dir+'eval/',self.model_name,('logP: '+str(self.best_validation_logs)),epoch)
            better = True
        
        if better:
            self.test(test_data,epoch=epoch)

        
        write_out(self.stats_log_dir,self.model_name,'Overall: '+str(self.best_validation_overall)+
                ' logD:'+str(np.round(logD_loss,4))+
                ' logS:'+str(np.round(logS_loss,4))+
                ' logP:'+str(np.round(logP_loss,4))+
                ' epoch:'+str(epoch)+
                self.model_name)
        with val_summary_writer.as_default():
            tf.summary.scalar('eval OVERALL loss', overall_evaluation_loss, step=epoch)
            if self.config.include_logD:
                tf.summary.scalar('eval logD loss', logD_loss, step=epoch)
            if self.config.include_logP:
                tf.summary.scalar('eval logP loss', logP_loss, step=epoch)
            if self.config.include_logS:
                tf.summary.scalar('eval logS loss', logS_loss, step=epoch)
        better = False
        logging.debug("Evaluation Ends!")

    def test(self, test_data,test_data_ml=None,epoch=None,include_dropout=False,test_n_times=1,
            also_evaluate=False,eval_data=None,consensus=False,plot_reg_result=False,
            ml_models=None,stds=None):
        '''
        evaluates the trained model. The results are being saved in tensorboard files
        as well as in text files.\n
        INPUT:\n
            test_data (list of AtomGraphs): test data consisting of AtomGraph instances for
            the GNN methods \n
            test_data_ml (list of (representation, properties)): used to train the non-GNN models. 
                The representation is based on which fingerprints and or descriptors are being used.
                The properties are the same as for the gnn methods. All are in the following
                order: logd, logs, lop! \n
            epoch (int): current epoch - when statistics are writte, to see which epoch is there \n
            include_dropout (bool): Include dropout during testing can be seen as generating 
                confidence intervalls \n
            test_n_times (int): number of bootstrap iterations \n
            also_evaluate (bool): do you also want to use the evaluation data to see in 
                the statistics, where it stands (easier to compare) \n
            eval_data (list of AtomGraphs): evaluation set - if 'also_evaluate' is True \n
            consensus (bool): if True, the non-GNN models are also trained and tested and an
                consensus scoring including the GNN and non-GNN is calculated \n
            plot_reg_result (bool): regression plot of the model in the 'plot_dir' under 'model_name' \n
            ml_models (Sklearn ML Models): used to test the non-GNN models. List of models
            trained on logd, logs, logp \n
            stds (Sklearn ML Standardizer): used to standardize the representations in the testing
            similar as during training. list of logd, logs, logp stds\n
        RETURNS:\n
            None \n 
        '''
        logging.debug("Test starts!")
        logging.debug("Check weather consensus and training data are defined")
        if (consensus and ml_models) is None:
            logging.error("Cannot do consensus scoring without training data! Please define!")
            print("Cannot do consensus scoring without training data! Please define!")
            return None
        #####
        logD_loss = []
        logP_loss = []
        logS_loss = []
        logD_r2 = []
        logP_r2 = []
        logS_r2 = []

        logD_loss_cons = list()
        logP_loss_cons = list()
        logS_loss_cons = list()
        logD_r2_cons = []
        logP_r2_cons = []
        logS_r2_cons = []
        ml_result_dict = dict()
        #####
        logging.debug("Start GNN MODELS TESTING!")
        for i in range(0,test_n_times):
            # use for plot and statistics (gather_*) for each log endpoint individually.
            # *_exp are for the true values; *_pred are for the predictions.
            plot_logD_exp,plot_logD_pred,gather_logD_pred,gather_logD_exp = list(),list(),list(),list()
            plot_logP_exp,plot_logP_pred,gather_logP_pred,gather_logP_exp = list(),list(),list(),list()
            plot_logS_exp,plot_logS_pred,gather_logS_pred,gather_logS_exp = list(),list(),list(),list()
       
            gather_logD_pred_ml = list()
            gather_logP_pred_ml = list()
            gather_logS_pred_ml = list()
            # random.seed(0)
            samples = random.sample(range(0, len(test_data)), len(test_data)-15)
            # batches = copy.copy(test_data)
            # batches.pop(i)
            test_data_boot = list()
            if test_n_times > 1:
                for j in samples:
                    test_data_boot.append(test_data[j])
            else:
                test_data_boot = test_data
            for batch in test_data_boot:
                for instance in batch:
                    # try:
                    predictions,dgin_encoding = self.call(instance,train=include_dropout)
                    
                    ## logD prediction - for plot and current losses
                    if self.config.include_logD:
                        if instance.properties['logd']:
                            gather_logD_exp.append(instance.properties['logd'])
                            gather_logD_pred.append(predictions['logD_pred'][0][0].numpy())
                            plot_logD_exp.append(instance.properties['logd']) # for plotting - 1.6
                            plot_logD_pred.append(predictions['logD_pred'][0][0].numpy())# so it starts at 0

                    ## logS prediction - for plot and current losses
                    if self.config.include_logS:
                        if instance.properties['logs']:
                            if self.config.include_logD:
                                if not (str(instance.properties['logs']) == str(7.95)):
                                    if not (str(np.round(instance.properties['logs'],4)) == str(5.37)):
                                        gather_logS_exp.append(instance.properties['logs'])
                                        gather_logS_pred.append(predictions['logS_pred'][0][0].numpy())
                                        plot_logS_exp.append(instance.properties['logs'])
                                        plot_logS_pred.append(predictions['logS_pred'][0][0].numpy())
                            else:
                                gather_logS_exp.append(instance.properties['logs'])
                                gather_logS_pred.append(predictions['logS_pred'][0][0].numpy())
                                plot_logS_exp.append(instance.properties['logs'])
                                plot_logS_pred.append(predictions['logS_pred'][0][0].numpy())

                    ## logP prediction - for plot and current losses
                    if self.config.include_logP:
                        gather_logP_exp.append(instance.properties['logp'])
                        gather_logP_pred.append(predictions['logP_pred'][0][0].numpy())
                        plot_logP_exp.append(instance.properties['logp'])
                        plot_logP_pred.append(predictions['logP_pred'][0][0].numpy())

                # except Exception as exc:
                #     print("Well well well, some issues during testing...",exc)
                #     continue
            ############## consensus ML
            if consensus:
                test_representations, test_properties = test_data_ml[0],test_data_ml[1]
                ####only the best ML models with the best AI models as consensus
                if self.config.include_logD:
                    batched_repr = make_batch(test_representations[0],self.batch_size)
                    test_data_boot_ml = list()
                    print("test_representations[0]",len(test_representations[0]))
                    print("batched_repr",len(batched_repr))
                    print("samples",samples)
                    if test_n_times > 1:
                        for j in samples:
                            if j < 28:
                                test_data_boot_ml.append(batched_repr[j])
                    else:
                        test_data_boot_ml = batched_repr
                    print("test_data_boot_ml logD",len(test_data_boot_ml))
                    print("gather_logD_pred_ml",tf.shape(gather_logD_pred))
                    print("gather_logD_exp",tf.shape(gather_logD_exp))
                    gather_logD_pred_ml = test_model(test_data_boot,ml_models[0],stds[0])
                    try:
                        from operator import add
                        cons = list( map(add, gather_logD_pred_ml, gather_logD_pred) )
                        cons = np.divide(cons,2)
                        logD_loss_cons.append(tf.math.sqrt(self.lipo_loss_mse(gather_logD_exp,cons)))
                        logD_r2_cons.append(r2_score(gather_logD_exp,cons))
                    except Exception as e:
                        print("error during logD testing:",e)
                    
                if self.config.include_logS:
                    batched_repr = make_batch(test_representations[1],self.batch_size)
                    test_data_boot_ml = list()
                    if test_n_times > 1:
                        for j in samples:
                            if j > 27:
                                j = j - 27
                                test_data_boot_ml.append(batched_repr[j])
                    else:
                        test_data_boot_ml = batched_repr
                    print("test_data_boot_ml logS",len(test_data_boot_ml))
                    print("gather_logD_pred_ml",tf.shape(gather_logS_pred))
                    gather_logS_pred_ml = test_model(test_data_boot_ml,ml_models[1],stds[1])
                    try:
                        from operator import add
                        cons = list( map(add, gather_logS_pred_ml, gather_logS_pred) )
                        cons = np.divide(cons,2)
                        logS_loss_cons.append(tf.math.sqrt(self.logS_loss_mse(gather_logS_exp,cons)))
                        logS_r2_cons.append(r2_score(gather_logS_exp,cons))
                    except Exception as e:
                        print("error during logS testing:",e)

                if self.config.include_logP:
                    batched_repr = make_batch(test_representations[2],self.batch_size)
                    test_data_boot_ml = list()
                    if test_n_times > 1:
                        for j in samples:
                            test_data_boot_ml.append(batched_repr[j])
                    else:
                        test_data_boot_ml = batched_repr
                    gather_logP_pred_ml = test_model(test_data_boot_ml,ml_models[2],stds[2])
                    try:
                        from operator import add
                        cons = list( map(add, gather_logP_pred_ml, gather_logP_pred) )
                        cons = np.divide(cons,2)
                        logP_loss_cons.append(tf.math.sqrt(self.logP_loss_mse(gather_logP_exp,cons)))
                        logP_r2_cons.append(r2_score(gather_logP_exp,cons))
                    except Exception as e:
                        print("error during logP testing:",e)
                    
            # logD
            if self.config.include_logD:
                logD_loss.append(tf.math.sqrt(self.lipo_loss_mse(gather_logD_exp,gather_logD_pred)))
            # logP
            if self.config.include_logP:
                logP_loss.append(tf.math.sqrt(self.logP_loss_mse(gather_logP_exp,gather_logP_pred)))
            # logS
            if self.config.include_logS:
                logS_loss.append(tf.math.sqrt(self.logS_loss_mse(gather_logS_exp,gather_logS_pred)))
            ### overall
            df = pd.DataFrame()
            if self.config.include_logD:
                df['predicted'] = plot_logD_pred
                df['experimental'] = plot_logD_exp
                logD_r2.append(get_r2(df))
            
            if self.config.include_logS:
                df = pd.DataFrame()
                df['predicted'] = plot_logS_pred
                df['experimental'] = plot_logS_exp
                logS_r2.append(get_r2(df))

            if self.config.include_logP:
                df = pd.DataFrame()
                df['predicted'] = plot_logP_pred
                df['experimental'] = plot_logP_exp
                logP_r2.append(get_r2(df))

        logS_loss_single,lower_lim_logS,upper_lim_logS = 0.0,0.0,0.0
        logP_loss_single,lower_lim_logP,upper_lim_logP = 0.0,0.0,0.0
        logD_loss_single,lower_lim_logD,upper_lim_logD = 0.0,0.0,0.0

        logS_loss_single_cons,lower_lim_logS_cons,upper_lim_logS_cons = 0.0,0.0,0.0
        logP_loss_single_cons,lower_lim_logP_cons,upper_lim_logP_cons = 0.0,0.0,0.0
        logD_loss_single_cons,lower_lim_logD_cons,upper_lim_logD_cons = 0.0,0.0,0.0

        #r2
        logS_r2_single,lower_lim_logS_r2,upper_lim_logS_r2 = 0.0,0.0,0.0
        logP_r2_single,lower_lim_logP_r2,upper_lim_logP_r2 = 0.0,0.0,0.0
        logD_r2_single,lower_lim_logD_r2,upper_lim_logD_r2 = 0.0,0.0,0.0

        logS_r2_single_cons,lower_lim_logS_r2_cons,upper_lim_logS_r2_cons = 0.0,0.0,0.0
        logP_r2_single_cons,lower_lim_logP_r2_cons,upper_lim_logP_r2_cons = 0.0,0.0,0.0
        logD_r2_single_cons,lower_lim_logD_r2_cons,upper_lim_logD_r2_cons = 0.0,0.0,0.0

        ci = 0.95
        if self.config.include_logD:
            logD_loss_single = tf.math.reduce_mean(logD_loss).numpy()
            logD_r2_single = tf.math.reduce_mean(logD_r2).numpy()
            if test_n_times > 1:
                lower_lim_logD,upper_lim_logD = np.quantile(logD_loss, [0.025,0.025+ci], axis=0)
                #r2
                lower_lim_logD_r2,upper_lim_logD_r2 = np.quantile(logD_r2, [0.025,0.025+ci], axis=0)
                #####
            if consensus:
                #### loss
                logD_loss_single_cons = tf.math.reduce_mean(logD_loss_cons).numpy()
                lower_lim_logD_cons,upper_lim_logD_cons = np.quantile(logD_loss_cons, [0.025,0.025+ci], axis=0)
                #### r2
                logD_r2_single_cons = tf.math.reduce_mean(logD_r2_cons).numpy()
                lower_lim_logD_r2_cons,upper_lim_logD_r2_cons = np.quantile(logD_r2_cons, [0.025,0.025+ci], axis=0)
            ####
        if self.config.include_logS:
            logS_loss_single = tf.math.reduce_mean(logS_loss).numpy()
            logS_r2_single = tf.math.reduce_mean(logS_r2).numpy()
            if test_n_times > 1:
                lower_lim_logS,upper_lim_logS = np.quantile(logS_loss, [0.025,0.025+ci], axis=0)
                #r2
                lower_lim_logS_r2,upper_lim_logS_r2 = np.quantile(logS_r2, [0.025,0.025+ci], axis=0)
            if consensus:
                #### loss
                logS_loss_single_cons = tf.math.reduce_mean(logS_loss_cons).numpy()
                lower_lim_logS_cons,upper_lim_logS_cons = np.quantile(logS_loss_cons, [0.025,0.025+ci], axis=0)
                #### r2
                logS_r2_single_cons = tf.math.reduce_mean(logS_r2_cons).numpy()
                lower_lim_logS_r2_cons,upper_lim_logS_r2_cons = np.quantile(logS_r2_cons, [0.025,0.025+ci], axis=0)
        if self.config.include_logP:
            logP_loss_single = tf.math.reduce_mean(logP_loss).numpy()
            logP_r2_single = tf.math.reduce_mean(logP_r2).numpy()
            if test_n_times > 1:
                lower_lim_logP,upper_lim_logP = np.quantile(logP_loss, [0.025,0.025+ci], axis=0)
                #r2
                lower_lim_logP_r2,upper_lim_logP_r2 = np.quantile(logP_r2, [0.025,0.025+ci], axis=0)
            if consensus:
                #### loss
                logP_loss_single_cons = tf.math.reduce_mean(logP_loss_cons).numpy()
                lower_lim_logP_cons,upper_lim_logP_cons = np.quantile(logP_loss_cons, [0.025,0.025+ci], axis=0)
                #### r2
                logP_r2_single_cons = tf.math.reduce_mean(logP_r2_cons).numpy()
                lower_lim_logP_r2_cons,upper_lim_logP_r2_cons = np.quantile(logP_r2_cons, [0.025,0.025+ci], axis=0)

        ### overall
        overall_loss = logD_loss_single+logP_loss_single+logS_loss_single
        ##############
        eval_logD_loss_single,eval_logS_loss_single,eval_logP_loss_single = 0,0,0
        if also_evaluate:
            eval_logD_loss_single,eval_logS_loss_single,eval_logP_loss_single = self.inter_evaluate(eval_data)
        ##############
        if consensus:
            write_out(self.stats_log_dir+'test/','best_test_consensus_all',
                ' logD:'+str(np.round(logD_loss_single_cons,4))+' lowD:'+str(np.round(lower_lim_logD_cons,4))+' upD:'+str(np.round(upper_lim_logD_cons,4))+
                ' logD_r2:'+str(np.round(logD_r2_single_cons,4))+' lowD_r2:'+str(np.round(lower_lim_logD_r2_cons,4))+' upD_r2:'+str(np.round(upper_lim_logD_r2_cons,4))+
                ' logS:'+str(np.round(logS_loss_single_cons,4))+' lowS:'+str(np.round(lower_lim_logS_cons,4))+' upS:'+str(np.round(upper_lim_logS_cons,4))+
                ' logS_r2:'+str(np.round(logS_r2_single_cons,4))+' lowS_r2:'+str(np.round(lower_lim_logS_r2_cons,4))+' upS_r2:'+str(np.round(upper_lim_logS_r2_cons,4))+
                ' logP:'+str(np.round(logP_loss_single_cons,4))+' lowP:'+str(np.round(lower_lim_logP_cons,4))+' upP:'+str(np.round(upper_lim_logP_cons,4))+
                ' logP_r2:'+str(np.round(logP_r2_single_cons,4))+' lowP_r2:'+str(np.round(lower_lim_logP_r2_cons,4))+' upP_r2:'+str(np.round(upper_lim_logP_r2_cons,4))+
                ' epoch:'+str(epoch)+' n_predict_boot:'+str(test_n_times)+' consensus:'+str(1),
                self.model_name)

        write_out(self.stats_log_dir+'test/','best_test_all',
                ' logD:'+str(np.round(logD_loss_single,4))+' lowD:'+str(np.round(lower_lim_logD,4))+' upD:'+str(np.round(upper_lim_logD,4))+
                ' logD_r2:'+str(np.round(logD_r2_single,4))+' lowD_r2:'+str(np.round(lower_lim_logD_r2,4))+' upD_r2:'+str(np.round(upper_lim_logD_r2,4))+
                ' logS:'+str(np.round(logS_loss_single,4))+' lowS:'+str(np.round(lower_lim_logS,4))+' upS:'+str(np.round(upper_lim_logS,4))+
                ' logS_r2:'+str(np.round(logS_r2_single,4))+' lowS_r2:'+str(np.round(lower_lim_logS_r2,4))+' upS_r2:'+str(np.round(upper_lim_logS_r2,4))+
                ' logP:'+str(np.round(logP_loss_single,4))+' lowP:'+str(np.round(lower_lim_logP,4))+' upP:'+str(np.round(upper_lim_logP,4))+
                ' logP_r2:'+str(np.round(logP_r2_single,4))+' lowP_r2:'+str(np.round(lower_lim_logP_r2,4))+' upP_r2:'+str(np.round(upper_lim_logP_r2,4))+
                ' eval_logD:'+str(np.round(eval_logD_loss_single,4))+
                ' eval_logS:'+str(np.round(eval_logS_loss_single,4))+
                ' eval_logP:'+str(np.round(eval_logP_loss_single,4))+
                ' epoch:'+str(epoch)+' n_predict_boot:'+str(test_n_times)+' consensus:'+str(0),
                self.model_name)

        # ### not necessary for current approach - only when testing; takes up too much memory
        # if plot_reg_result:
        #     df = pd.DataFrame()
        #     if self.config.include_logD:
        #         df['predicted'] = plot_logD_pred
        #         df['experimental'] = plot_logD_exp
        #         df['names'] = plot_names_logD
        #         plot((df,'lin_reg',self.plot_dir+'LOGD_plot_epoch_'+str(epoch)+'_dd_'+str(logD_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'))
        #         plot((df,'res_den',self.plot_dir+'LOGD_resi_plot_epoch_'+str(epoch)+str(logD_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'))
            
        #     if self.config.include_logS:
        #         df = pd.DataFrame()
        #         df['predicted'] = plot_logS_pred
        #         df['experimental'] = plot_logS_exp
        #         df['names'] = plot_names_logS
        #         plot((df,'lin_reg',self.plot_dir+'LOGS_plot_epoch_'+str(epoch)+str(logS_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'),xlim=10,ylim=9.5)
        #         plot((df,'res_den',self.plot_dir+'LOGS_resi_plot_epoch_'+str(epoch)+str(logS_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'),xlim=10,ylim=9.5)

        #     if self.config.include_logP:
        #         df = pd.DataFrame()
        #         df['predicted'] = plot_logP_pred
        #         df['experimental'] = plot_logP_exp
        #         df['names'] = plot_names_logP
        #         plot((df,'lin_reg',self.plot_dir+'LOGP_plot_epoch_'+str(epoch)+str(logP_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'),xlim=10,ylim=9.5)
        #         plot((df,'res_den',self.plot_dir+'LOGP_resi_plot_epoch_'+str(epoch)+str(logP_loss.numpy())+'_'+str(overall_loss.numpy())+'.png'),xlim=10,ylim=9.5)
