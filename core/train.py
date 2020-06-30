
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pandas as pd
from tqdm import tqdm
from time import sleep

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
    pva_multi['actual'] = self.test_target
    for i in tqdm(range(0, n), desc="Model Replication"):  # run model n times
        start_time = time()
        if self.algorithm == 'nn':
            self.regressor.fit(self.train_features, self.train_target, **self.fit_params)

        else:
            self.regressor.fit(self.train_features, self.train_target)

        # Make predictions
        predictions = self.regressor.predict(self.test_features)
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

    pva_multi['pred_avg'] = pva.mean(axis=1)
    pva_multi['pred_std'] = pva.std(axis=1)

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
    self.predictions = pva_multi
    self.predictions_stats = stats

    print('Average R^2 = %.3f' % stats['r2_avg'], '+- %.3f' % stats['r2_std'])
    print('Average RMSE = %.3f' % stats['rmse_avg'], '+- %.3f' % stats['rmse_std'])
    print()

def train_cls(self, n=5):
    # set the model specific classifier function from sklearn

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)
    acc = np.empty(n)
    #conf = np.empty(n)
    #clsrep = np.empty(n)
    auc = np.empty(n)
    t = np.empty(n)

    cls_multi = pd.DataFrame([])
    cls_multi['actual'] = self.test_target
    for i in tqdm(range(0, n), desc="Model Replication"):  # run model n times
        start_time = time()
        self.regressor.fit(self.train_features, self.train_target)

        # Make predictions
        predictions = self.regressor.predict(self.test_features)
        done_time = time()
        fit_time = done_time - start_time

        # Dataframe for replicate_model
        cls = pd.DataFrame([], columns=['actual', 'predicted'])
        cls['actual'] = self.test_target
        cls['predicted'] = predictions
        acc[i] = accuracy_score(cls['actual'], cls['predicted'])
        print()
        print('Accuracy score for this run: ')
        print(acc[i])
        # TODO fix confusion matrix and classificaiton report metrics
        conf = confusion_matrix(cls['actual'], cls['predicted'])
        print('Confusion matrix for this run: ')
        print(conf)
        clsrep = classification_report(cls['actual'], cls['predicted'])
        print('Classification report for this run: ')
        print(clsrep)
        auc[i] = roc_auc_score(cls['actual'], cls['predicted'])
        print('Roc_Auc score for this run: ')
        print(auc[i])
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
        # 'conf_avg': conf.mean(),
        # 'conf_std': conf.std(),
        'clsrep_raw': clsrep,
       # 'clsrep_avg': clsrep.mean(),
       # 'clsrep_std': clsrep.std(),
        'auc_raw': auc,
        'auc_avg': auc.mean(),
        'auc_std': auc.std(),
        'time_raw': t,
        'time_avg': t.mean(),
        'time_std': t.std()
    }
    self.predictions = cls_multi
    self.predictions_stats = stats
