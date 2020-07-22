
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
from tqdm import tqdm
from time import sleep


def train_reg(model, n=5):
    """
    Function to train the model n times and collect basic statistics about results.
    :param model:
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
    pva_multi['smiles'] = model.test_molecules  # Save smiles to predictions
    pva_multi['actual'] = model.test_target
    for i in tqdm(range(0, n), desc="Model Replication\n"):  # run model n times
        start_time = time()

        if model.algorithm == 'nn':  # needs fit_params to set epochs and callback
            model.regressor.fit(model.train_features, model.train_target, **model.fit_params)
        else:
            model.regressor.fit(model.train_features, model.train_target)

        # Make predictions
        predictions = model.regressor.predict(model.test_features)
        done_time = time()
        fit_time = done_time - start_time

        # Dataframe for replicate_model
        pva = pd.DataFrame([], columns=['actual', 'predicted'])
        pva['actual'] = model.test_target
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
    model.predictions = pva_multi
    model.predictions_stats = stats

    print('Average R^2 = %.3f' % stats['r2_avg'], '+- %.3f' % stats['r2_std'])
    print('Average RMSE = %.3f' % stats['rmse_avg'], '+- %.3f' % stats['rmse_std'])
    print()

    return model.predictions, model.predictions_stats


def train_cls(model, n=5):
    # set the model specific classifier function from sklearn

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)
    acc = np.empty(n)
    conf = np.zeros((n,2,2))
    #clsrep = np.empty(n)
    auc = np.empty(n)
    t = np.empty(n)
    f1_score1 = np.empty(n)

    cls_multi = pd.DataFrame([])
    cls_multi['smiles'] = model.test_molecules  # Save smiles to predictions
    if model.dataset not in ['sider.csv', 'clintox.csv']:
        cls_multi['actual'] = model.test_target  # Broken for mult-label classification
    for i in tqdm(range(0, n), desc="Model Replication"):  # run model n times
        print()
        start_time = time()
        model.regressor.fit(model.train_features, model.train_target)

        # Make predictions
        predictions = model.regressor.predict(model.test_features)
        done_time = time()
        fit_time = done_time - start_time

        # Contains multi-label evaluation methods
        if model.dataset in ['sider.csv', 'clintox.csv']:
            conf = multilabel_confusion_matrix(model.test_target, predictions)
            print("Confusion matrix for each individual label (see key above, labeled as 'target(s):'): ")
            print(conf)
            print()

            RF_pred = cross_val_predict(model.regressor, model.train_features, model.train_target, cv=3)
            f1_score1[i] = f1_score(model.train_target, RF_pred, average="macro")

            print()
            auc[i] = roc_auc_score(model.test_target, predictions)
            sleep(0.25)

        else:  # Contains single-label evaluation methods
            # Dataframe for replicate_model
            cls = pd.DataFrame([], columns=['actual', 'predicted'])
            cls['actual'] = model.test_target
            cls['predicted'] = predictions
            acc[i] = accuracy_score(cls['actual'], cls['predicted'])

            # TODO fix confusion matrix and classificaiton report metrics
            conf[i] = confusion_matrix(cls['actual'], cls['predicted'])
            print()
            print('Confusion matrix for this run: ')
            print(conf[i])

            clsrep = classification_report(cls['actual'], cls['predicted'])
            print('Classification report for this run: ')
            print(clsrep)
            auc[i] = roc_auc_score(cls['actual'], cls['predicted'])
            t[i] = fit_time

            # store as enumerated column for multipredict
            cls_multi['predicted' + str(i)] = predictions
            sleep(0.25)  # so progress bar can update

            # Define pred_avg and pred_std
            cls_multi['pred_avg'] = cls.mean(axis=1)
            cls_multi['pred_std'] = cls.std(axis=1)

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

    model.predictions = cls_multi
    model.predictions_stats = stats

    if model.dataset in ['sider.csv', 'clintox.csv']:
        print('Average roc_auc score = %.3f' % stats['auc_avg'], '+- %.3f' % stats['auc_std'])
        print('Average f1_score = %.3f' % stats['f1_score_avg'], '+- %.3f' % stats['f1_score_std'])
        print()
    else:
        print('Average accuracy score = %.3f' % stats['acc_avg'], '+- %.3f' % stats['acc_std'])
        print('Average roc_auc score = %.3f' % stats['auc_avg'], '+- %.3f' % stats['auc_std'])
        print()

    return model.predictions, model.predictions_stats
