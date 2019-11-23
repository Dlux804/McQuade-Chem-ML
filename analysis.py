import matplotlib.pyplot as plt
import numpy as np

def predict(regressor, train_features, test_features, train_target, test_target):
    """Fit model and predict target values.  Return data frame of actual and predicted
    values as well as model fit time."""
    fit_time = time()
    regressor.fit(train_features, train_target)
    done_time = time()
    fit_time = done_time - fit_time
    print('Finished Training After ',(done_time-fit_time),"sec\n")
    # Make predictions
    predictions = regressor.predict(test_features)  # Val predictions

    true = test_target
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = true
    pva['predicted'] = predictions
#    pva.to_csv(exp+expt+'-pva_data.csv')

    return pva, fit_time

def replicate_model(df, params, n, title, feat):
    """Run model n times.  Take average and standar deviation of resulting metrics. """

    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    for i in range(0,n): # run model n times
        train_features, test_features, train_target, test_target, feature_list = targets_features(df, random=None)
        pva = gdb_train_predict(params, train_features, test_features, train_target,
                                            test_target,
                                            title, feat)

        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))

    # TODO: Should I store these results in a dictonary?
    r2_avg = r2.mean()
    r2_std = r2.std()
    mse_avg = mse.mean()
    mse_std = mse.std()
    rmse_avg = rmse.mean()
    rmse_std = rmse.std()
    print('Average R^2 = %.3f' % r2_avg, '+- %.3f' % r2_std)
    print('Average RMSE = %.3f' % rmse_avg, '+- %.3f' % rmse_std)
    print()
    return r2_avg, r2_std, mse_avg, mse_std, rmse_avg, rmse_std


def pva_graphs(pva,model_name):
    """ Creates Predicted vs. Actual graph from predicted data. """
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
    print('R^2 = %.3f' % r2)
    print('MSE = %.3f' % mse)
    print('RMSE = %.3f' % rmse)

    plt.rcParams['figure.figsize']= [15,9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    plt.plot(pva['actual'], pva['predicted'], 'o')
    # ax = plt.axes()
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title('EXP:')  # TODO: Update naming scheme
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]
    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    plt.savefig(model_name+'-' +'.png')
    plt.show()
    return fig  # Can I store a graph as an attribute to a model?
