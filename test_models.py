import pandas as pd
import models
import analysis




def test_models_run(mocker):
    """

    :param mocker:
    :return:
    """
    mocker.patch('models.MlModel.run')
    # Using mocker to replace the class method with a no-op object that can be called but does not do anything and typically has no side effects
    # print(mock_analysis_predict)
    model1 = models.MlModel('rf', 'water-energy.csv', 'expt')  # Calling MlModel to get our class instances
    assert type(model1.data) == pd.DataFrame  # Testing to see if data is a dataframe
    model1.featurization([0])
    model1.run(tune=False)
    models.MlModel.run.assert_called_once_with(tune=False)
