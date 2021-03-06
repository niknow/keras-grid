import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
from keras.models import Model, Sequential, load_model


class ModelGrid(ABC):
    """
    Abstract class to wrap a grid of keras models parametrized by a dictionary.
    """

    _history_suffix = '_history.pickle'
    _parameter_suffix = '_parameters.pickle'

    def __init__(self, parameter_dict, hyperparameter_dict):
        """
        :param parameter_dict:  Dictionary of parameters that the model. The keys index the parameters and the values
                                are the parameters used to construct the various models.
        :param hyperparameter_dict: Dictionary of hyperparameters, i.e. parameters which are the same for all of the
                                    models.
        """
        self.parameter_dict = parameter_dict
        self.hyperparameter_dict = hyperparameter_dict
        self.models = {}
        self.history = {}

    @classmethod
    def from_parameter_list(cls, parameter_list, hyperparameter_dict):
        """
        Factory to create an instance of the class via a list of parameter arrays instead of a dictionary.
        Resulting grid is built from dictionary combining all parameters in all the lists with each other.

        :param parameter_list: A list of 1D numpy arrays.
        :param hyperparameter_dict: The dict of hyperparameters passed to the constructor.
        :return: A ModelGrid instance where the parameter_dict is built as the cartesian product of all the
                parameters in the 1D numpy arrays of the list.
        """
        shapes = [a.shape[0] for a in parameter_list]
        return cls({key: np.array([param[i] for i, param in zip(key, parameter_list)])
                    for key, _ in np.ndenumerate(np.zeros(shapes))},
                   hyperparameter_dict)

    @classmethod
    def from_disk(cls, path, custom_objects=None):
        """
        Loads the trained models, parameters, and history from disk.
        """
        # load parameter dict
        with open(os.path.join(path, cls.__name__ + ModelGrid._parameter_suffix), 'rb') as fp:
            parameter_dict, hyperparameter_dict = pickle.load(fp)
            model_grid = cls(parameter_dict, hyperparameter_dict)
        for key in model_grid.parameter_dict:
            # load models
            print("Loading model %s..." % cls.key_to_string(key), end=' ')
            model_grid.models[key] = load_model(os.path.join(path, cls.__name__ + '_%s.h5' % cls.key_to_string(key)),
                                                custom_objects=custom_objects)
        # load history
        with open(os.path.join(path, cls.__name__ + ModelGrid._history_suffix), 'rb') as fp:
            model_grid.history = pickle.load(fp)
        return model_grid


    @classmethod
    def key_to_string(cls, key):
        """
        Converts the key to a string assuming the key is a tuple.
        """
        return '_'.join(map(str, key))

    @abstractmethod
    def _create_model(self, key):
        """
        This method needs to be implemented in any derived class that is not abstract. It creates a single model
        based on the parameters found in the dictionary under key.

        :param key: Key to find the model parameters in self.parameter_dict.
        :return: Instance of the model using the parameters.
        """
        pass

    def __getitem__(self, key):
        return self.models[key]

    def __setitem__(self, key, item):
        self.models[key] = item

    def initialize(self):
        """
        Creates all the models in the grid with the various parameters.
        """
        for key in self.parameter_dict:
            self.models[key] = self._create_model(key)

    def compile(self, *args, **kwargs):
        """
        Compiles all the models in the grid. All args and kwargs are passed to keras model `compile´ method.
        """
        for key in self.parameter_dict:
            self.models[key].compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Fits all the models in the grid to the same data. All args and kwargs are passed to keras model `fit´ method.
        :param args:    Parameters passed as arguments to `fit´ function of each keras model.
        :param kwargs:  Parameters passed as keyword arguments to `fit´ function of each keras model.
        :return: Dictionary with same keys as self.parameter_dict and values given by the result of `fit´ on each
                 associated model using args and kwargs.
        """
        for key in self.parameter_dict:
            print("Fitting with key %s " % type(self).key_to_string(key))
            self.history[key] = self.models[key].fit(*args, **kwargs).history
        return self.history

    def evaluate(self, *args, **kwargs):
        """
        Evaluates all the models in the grid. All args and kwargs are passed to keras model `evaluate´ method.
        :param args:    Parameters passed as arguments to `evaluate´ function of each keras model.
        :param kwargs:  Parameters passed as keyword arguments to `evaluate´ function of each keras model.
        :return: Dictionary with same keys as self.parameter_dict and values given by the result of `evaluate´ on each
                 associated model using args and kwargs.
        """
        evaluation = {}
        for key in self.parameter_dict:
            print(type(self).key_to_string(key))
            print("Evaluating model with key %s: " % type(self).key_to_string(key))
            evaluation[key] = self.models[key].evaluate(*args, **kwargs)
        return evaluation

    def predict(self, restrict_to_keys=None, *args, **kwargs):
        """
        Executes model prediction restricted to a subset of the models of the grid.

        :param restrict_to_keys: A list of keys to perform the prediction on. Default value is None, i.e. no restrictions.
        :param args:    Parameters passed as arguments to `predict´ function of each keras model.
        :param kwargs:  Parameters passed as keyword arguments to `predict´ function of each keras model.
        :return: Dictionary with keys given by restrict_to_keys and values given by the predictions of the associated
                 models using args and kwargs.
        """
        if restrict_to_keys is None:
            restrict_to_keys=self.parameter_dict.keys()
        predictions = {}
        for key in restrict_to_keys:
            print("Predicting for model with key %s: " % type(self).key_to_string(key))
            predictions[key] = self.models[key].predict(*args, **kwargs)
        return predictions

    def save(self, path):
        """
        Saves the model grid, all the model parameters and hyperparameters, as well as the training history
        into various files under path. It is recommended to use one folder for one model grid.
        """
        # create path if not exists
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        # save models
        for key in self.parameter_dict:
            self.models[key].save(os.path.join(path, type(self).__name__ + '_%s.h5' % type(self).key_to_string(key)))
        # save historys
        with open(os.path.join(path, type(self).__name__ + ModelGrid._history_suffix), 'wb') as fp:
            pickle.dump(self.history, fp)
        # save parameter and hyperparameter dict
        with open(os.path.join(path, type(self).__name__ + ModelGrid._parameter_suffix), 'wb') as fp:
            pickle.dump((self.parameter_dict, self.hyperparameter_dict), fp)
