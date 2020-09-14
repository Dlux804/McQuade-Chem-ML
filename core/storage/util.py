"""
Python file for utilities such as string handling. Mostly so we don't clog up misc.py
"""
from core.storage.dictionary import nn_default_param, get_param


def original_param(self):
    """
    Objective: Return untuned parameters that we care about. We want to acquire this data to compare the original
                parameter with tuned ones.
    :param self:
    :return:
    """
    param_dict = self.estimator.get_params()  # sklearn algorithms have a get_params() method that return their params
    new_param_dict = {}
    if self.algorithm == 'nn':  # If nn
        # Remove build_fn value. It returns a class object which is not what we want
        param_dict.pop('build_fn', None)
        # Update variable with NN default parameter stored in dictionary.py and defined in regressors.py
        param_dict.update(nn_default_param())
        self.original_param = param_dict  # Return original_param
    else:
        for key in param_dict:  # Iterate through keys in dictionary
            #  If the keys are not in
            if key in get_param(self.algorithm):
                new_param_dict[key] = param_dict[key]
                if key == 'base_estimator':
                    new_param_dict[key] = "None"
                # return new_param_dict
        self.original_param = new_param_dict  # Return original_parameter
