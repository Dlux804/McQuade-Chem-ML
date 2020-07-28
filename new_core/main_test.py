
from new_core.data import data
from new_core import models
from new_core.estimator.regressor import Regressor
from main import ROOT_DIR
import os
from new_core.utils import misc, name
from new_core.analyzer.train import Train
os.chdir(ROOT_DIR)  # Start in root directory

with misc.cd('dataFiles'):  # Initialize model
    model1 = models.MLModel(algorithm="gdb", dataset="ESOL.csv", target="water-sol", feat_meth=[0], tune=True,
                            cv_folds=2, opt_iter=2)
    data_loader = data.Data(dataset=model1.dataset, target=model1.target, feat_meth=model1.feat_meth)
    data_loader.featurize()
    data_loader.data_split()

    run_name = name.Name(model1).name()

    regressors = Regressor(algorithm=model1.algorithm, data_loader=data_loader, opt_iter=model1.opt_iter,
                           cv_folds=model1.cv, run_name=run_name)
    regressors.get_regressor(call=False)
    regressors.make_grid()
    regressors.hyperTune()

    analyer = Train(data_loader, regressors)
    analyer.train_reg()

    # regressors.get_regressor()
    # regressors.hyperTune()
    # ingestor = ingest.Ingest(model1.dataset, model1.target)
    #
    # ingestor.load_smiles()
    # featurizer = features.Feature(ingestor.data, model1.feat_meth)
    # featurizer.featurize()
    # splitter = split.Split(ingestor, featurizer.data, val=0.1)
    # splitter.data_split()
    # run_name = name.Name(model1).name()
    # model_grid = Grid(algorithm=model1.algorithm)
    # print(model1.cv)
    # regressors = Regressor(algorithm=model_grid.algorithm, splitter=splitter, opt_iter=model1.opt_iter, cv_folds=model1.cv, run_name=run_name)
    # regressors.get_regressor()
    # regressors.hyperTune()
