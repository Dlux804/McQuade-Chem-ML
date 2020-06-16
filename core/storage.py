import json
import pandas as pd
import numpy as np
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """
    Modifies JSONEncoder to convert numpy arrays to lists first.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
        print(k, type(v))
        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            objs.append(k)
            dfs.append(k)
            getattr(self, k).to_json(path_or_buf=self.run_name + '_' + k + '.json')

        if not isinstance(v, (int, float, dict, tuple, list, np.ndarray, bool, str, NoneType)):
            objs.append(k)


    objs = list(set(objs))
    print("Unsupported JSON Export Attributes:", objs)
    print("The following pandas attributes were exported to individual JSONs: ", dfs)
    for k in objs:
        del d[k]

    json_name = self.run_name + '_attributes' + '.json'
    with open(json_name, 'w') as f:
        # TODO fix param grid to be exported.  Current data type is unsupported
        # try:
        json.dump(d, f, cls=NumpyEncoder)
        # except TypeError:
        #     print("Unsupported JSON export data type.")
