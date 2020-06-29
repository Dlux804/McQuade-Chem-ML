import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


class NumpyEncoder(json.JSONEncoder):
    """
    Modifies JSONEncoder to convert numpy arrays to lists first.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def __clean_up_param_grid_item__(item):
    # This function will pull attributes of out Integer and Categorical classes generated from skopt
    d = dict(vars(item))  # Get attributes in sub-class
    labels_to_pop = []
    for label, item in d.items():
        try:
            dict(vars(item))  # Test if sub-sub attribute is also a class
            labels_to_pop.append(label)  # If so remove it
        except TypeError:
            pass
    for label in labels_to_pop:
        d.pop(label)
    return d


def export_json(self):
    """
    export entire model instance as a json file.  First converts all attributes to dictionary format.
    Some attributes will have unsupported data types for JSON export.  Notably, numpy arrays, Pandas
    Series and DataFrames and any other Python object, like graphs, molecule objects, etc.  Pandas
    objects are exported to their own JSON via pandas API.  All other unsupported objects are dropped.
    :param self:
    :return: writes json file
    """
    print("Model attributes are being exported to JSON format...")
    NoneType = type(None)  # weird but necessary decalartion for isinstance() to work on None
    d = dict(vars(self))  # create dict of model attributes
    objs = []
    dfs = []
    for k, v in tqdm(d.items(), desc="Export to JSON", position=0):

        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            getattr(self, k).to_json(path_or_buf=self.run_name + '_' + k + '.json')

        if not isinstance(v, (int, float, dict, tuple, list, np.ndarray, bool, str, NoneType)):
            objs.append(k)

        if k == 'param_grid':  # Param grid does not behave properly,
            new_param_grid_dict = {}
            for label, item in d[k].items():
                new_param_grid_dict[label] = __clean_up_param_grid_item__(item)  # Cleanup each item in param_gird dict
            d[k] = new_param_grid_dict

    objs = list(set(objs))
    print("Unsupported JSON Export Attributes:", objs)
    print("The following pandas attributes were exported to individual JSONs: ", dfs)
    for k in objs:
        del d[k]

    json_name = self.run_name + '_attributes' + '.json'
    with open(json_name, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder)


def pickle_model(model, file_location):  # Pickle models, at any stage of model
    with open(file_location, 'wb') as f:
        pickle.dump(model, f)


def unpickle_model(file_location):  # Unpickle model, re-run or continue running model
    with open(file_location, 'rb') as f:
        return pickle.load(f)
