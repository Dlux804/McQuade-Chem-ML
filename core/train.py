
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm

def train(self,n=5):
    """
    Function to train the model n times and collect basic statistics about results.
    :param self:
    :param n: number of replicates
    :return:
    """

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)
    # self.regressor.fit(self.train_features, self.train_target)
    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    t = np.empty(n)


    pva_multi = pd.DataFrame([])
    pva_multi['actual'] = self.test_target
    for i in tqdm(range(0, n)):  # run model n times
        start_time = time()
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
    print('done after {:.1f} seconds'.format(t.sum()))
    # done_time = time()
    # fit_time = done_time - start_time
    # t = fit_time
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
