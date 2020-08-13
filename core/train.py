
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
def train_reg(self,n=5):
    """
    Function to train the model n times and collect basic statistics about results.
    :param self:
    :param n: number of replicates
    :return:
    """

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)

    # empty arrays for storing replicated data
    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    t = np.empty(n)

    pva_multi = pd.DataFrame([])
    pva_multi['smiles'] = self.test_molecules  # Save smiles to predictions
    pva_multi['actual'] = self.test_target

    for i in tqdm(range(0, n), desc="Model Replication\n"):  # run model n times
        start_time = time()

        if self.algorithm == 'nn':  # needs fit_params to set epochs and callback
            self.estimator.fit(self.train_features, self.train_target, **self.fit_params)
        else:
            self.estimator.fit(self.train_features, self.train_target)

        # Make predictions
        predictions = self.estimator.predict(self.test_features)
        done_time = time()
        fit_time = done_time - start_time

        # Dataframe for replicate_model
        pva = pd.DataFrame([], columns=['actual', 'predicted'])
        pva['actual'] = self.test_target
        pva['predicted'] = predictions
        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
        t[i] = fit_time
        # store as enumerated column for multipredict
        pva_multi['predicted' + str(i)] = predictions
        sleep(0.25)  # so progress bar can update
    print('Done after {:.1f} seconds.'.format(t.sum()))
    # Store predicted columns and calculate predicted_average and predicted_std
    predicted_columns = pva_multi.columns.difference(['smiles', 'actual'])
    pva_multi['pred_avg'] = pva_multi[predicted_columns].mean(axis=1)
    pva_multi['pred_std'] = pva_multi[predicted_columns].std(axis=1)

    # Holding variables for scaled data
    scaled_r2 = np.empty(n)
    scaled_mse = np.empty(n)
    scaled_rmse = np.empty(n)

    # Drop smiles (and save them) then calculate max/min values of entire predicted dataframe
    smiles = pva_multi['smiles']
    pva_multi_scaled = pva_multi.drop('smiles', axis=1)
    data_max = max(pva_multi_scaled.max())  # Find abs min/max of predicted data
    data_min = min(pva_multi_scaled.min())

    # Logic to scale the predicted data
    pva_multi_scaled = (pva_multi_scaled - data_min) / (data_max - data_min)

    # Calculate r2, rmse, mse or for each pva columns
    for i, predicted_column in enumerate(predicted_columns):
        scaled_r2[i] = r2_score(pva_multi_scaled['actual'], pva_multi_scaled[predicted_column])
        scaled_mse[i] = mean_squared_error(pva_multi_scaled['actual'], pva_multi_scaled[predicted_column])
        scaled_rmse[i] = np.sqrt(scaled_mse[i])

    # Tack on info to marry pva_multi and pva_multi_scaled
    pva_multi_scaled['smiles'] = smiles
    pva_multi_scaled['pred_avg'] = pva_multi_scaled[predicted_columns].mean(axis=1)
    pva_multi_scaled['pred_std'] = pva_multi_scaled[predicted_columns].std(axis=1)

    stats = {
        'r2_raw': r2,
        'r2_avg': r2.mean(),
        'r2_std': r2.std(),
        'mse_raw': mse,
        'mse_avg': mse.mean(),
        'mse_std': mse.std(),
        'rmse_raw': rmse,
        'rmse_avg': rmse.mean(),
        'rmse_std': rmse.std(),
        'time_raw': t,
        'time_avg': t.mean(),
        'time_std': t.std()
    }

    scaled_stats = {
        'r2_raw_scaled': scaled_r2,
        'r2_avg_scaled': scaled_r2.mean(),
        'r2_std_scaled': scaled_r2.std(),
        'mse_raw_scaled': scaled_mse,
        'mse_avg_scaled': scaled_mse.mean(),
        'mse_std_scaled': scaled_mse.std(),
        'rmse_raw_scaled': scaled_rmse,
        'rmse_avg_scaled': scaled_rmse.mean(),
        'rmse_std_scaled': scaled_rmse.std(),
    }

    self.predictions = pva_multi
    self.predictions_stats = stats

    self.scaled_prediction = pva_multi_scaled
    self.scaled_predictions_stats = scaled_stats

    print('Average R^2 = %.3f' % stats['r2_avg'], '+- %.3f' % stats['r2_std'])
    print('Average RMSE = %.3f' % stats['rmse_avg'], '+- %.3f' % stats['rmse_std'])
    print()


def train_cls(self, n=5):
    # set the model specific classifier function from sklearn

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)
    acc = np.empty(n)
    conf = np.zeros((n,2,2))
    #clsrep = np.empty(n)
    auc = np.empty(n)
    t = np.empty(n)
    f1_score1 = np.empty(n)

    cls_multi = pd.DataFrame([])
    cls_multi['smiles'] = self.test_molecules  # Save smiles to predictions
#    cls_multi['actual'] = self.test_target
    for i in tqdm(range(0, n), desc="Model Replication"):  # run model n times
        print()
        start_time = time()
        self.estimator.fit(self.train_features, self.train_target)

        # Make predictions
        predictions = self.estimator.predict(self.test_features)
        done_time = time()
        fit_time = done_time - start_time
        if self.task_type == 'multi_label_classification': # Contains multi-label evaluation methods, since sider data set and clintox data set are done with multi-target classification.
            conf = multilabel_confusion_matrix(self.test_target, predictions)
            print("Confusion matrix for each individual label (see key above, labeled as 'target(s):'): ")
            print(conf)
            print()

            f1_score1[i] = f1_score(self.test_target, predictions, average="macro")

            print()
            auc[i] = roc_auc_score(self.test_target, predictions)
            sleep(0.25)
        else:  # Contains single-label evaluation methods
            # Dataframe for replicate_model
            cls = pd.DataFrame([], columns=['actual', 'predicted'])
            cls['actual'] = self.test_target
            cls['predicted'] = predictions

            acc[i] = accuracy_score(cls['actual'], cls['predicted'])


            f1_score1[i] = f1_score(cls['actual'], cls['predicted'], average="macro")


            conf[i] = confusion_matrix(cls['actual'], cls['predicted'])
            print()
            print('Confusion matrix for this run: ')
            print(conf[i])


            clsrep = classification_report(cls['actual'], cls['predicted'], output_dict=True)
            report = classification_report(cls['actual'], cls['predicted'])
            print('Classification report for this run: ')
            print(report)
            self.clsrepdf = pd.DataFrame(clsrep).transpose()


            auc[i] = roc_auc_score(cls['actual'], cls['predicted'])
            t[i] = fit_time

            # store as enumerated column for multipredict
            cls_multi['predicted' + str(i)] = predictions
            sleep(0.25)  # so progress bar can update

    print('Done after {:.1f} seconds.'.format(t.sum()))

    stats = {
        'acc_raw': acc,
        'acc_avg': acc.mean(),
        'acc_std': acc.std(),
        'conf_raw': conf,
        'conf_avg': conf.mean(),
        'conf_std': conf.std(),
       # 'clsrep_raw': clsrep,
        # 'clsrep_avg': clsrep.mean(),
        # 'clsrep_std': clsrep.std(),
        'auc_raw': auc,
        'auc_avg': auc.mean(),
        'auc_std': auc.std(),
        'time_raw': t,
        'time_avg': t.mean(),
        'time_std': t.std(),
        'f1_score_raw': f1_score1,
        'f1_score_avg': f1_score1.mean(),
        'f1_score_std': f1_score1.std()
    }
    self.predictions = predictions
    self.predictions_stats = stats
    self.auc_avg = stats['auc_avg']
    self.acc_avg = stats['acc_avg']
    self.f1_score_avg = stats['f1_score_avg']

    if self.task_type == 'multi_label_classification':
        print('Average roc_auc score = %.3f' % stats['auc_avg'], '+- %.3f' % stats['auc_std'])
        print('Average f1_score = %.3f' % stats['f1_score_avg'], '+- %.3f' % stats['f1_score_std'])
        print()
    else:
        print('Average accuracy score = %.3f' % stats['acc_avg'], '+- %.3f' % stats['acc_std'])
        print('Average roc_auc score = %.3f' % stats['auc_avg'], '+- %.3f' % stats['auc_std'])
        print('Average f1_score = %.3f' % stats['f1_score_avg'], '+- %.3f' % stats['f1_score_std'])
        print()

