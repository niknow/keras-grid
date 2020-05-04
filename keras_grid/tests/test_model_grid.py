import os
from unittest import TestCase
import numpy as np
import tempfile

from keras.models import Sequential
from keras.layers import Dense

from keras_grid.model_grid import ModelGrid

class MlpGrid(ModelGrid):
    """
    This class wraps a 2D grid of keras sequential models with dense layers.
    It assumes that parameters[i,j] is
    """

    def _create_model(self, key):
        num_inputs = self.hyperparameter_dict['num_inputs']
        num_outputs = self.hyperparameter_dict['num_outputs']
        num_layers, num_units = self.parameter_dict[key]
        model = Sequential()
        model.add(Dense(units=num_units, input_shape=(num_inputs,), activation='sigmoid'))
        for layer in range(1, num_layers - 1):
            model.add(Dense(units=num_units, activation='sigmoid'))
        model.add(Dense(units=num_outputs, activation='linear'))
        return model


class TestModelGrid(TestCase):
    """
    Tests abstract ModelGrid class via MLPGrid instantiation.
    """

    def setUp(self):
        self.parameter_dict = {(i, j): (layer, units)
                               for i, layer in enumerate([2, 3])
                               for j, units in enumerate([16, 32, 64])}
        self.hyperparameter_dict  = {
            'num_inputs': 3,
            'num_outputs': 4
        }
        self.mlpg = MlpGrid(self.parameter_dict, self.hyperparameter_dict)

    def test_can_instantiate(self):
        self.assertTrue(isinstance(self.mlpg, MlpGrid))

    def test_initialize(self):
        self.mlpg.initialize()
        self.assertEqual(self.mlpg.models.keys(), self.parameter_dict.keys())

    def test_compile(self):
        self.mlpg.initialize()
        self.mlpg.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')
        self.assertEqual(self.mlpg.models.keys(), self.parameter_dict.keys())

    def test_fit(self):
        self.mlpg.initialize()
        self.mlpg.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')
        self.mlpg.fit(x=np.random.normal(0, 1, (10, 3)),
                      y=np.random.normal(0, 1, (10, 4)),
                      epochs=1)
        self.assertEqual(self.mlpg.history.keys(), self.parameter_dict.keys())

    def test_evaluate(self):
        self.mlpg.initialize()
        self.mlpg.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')
        self.evaluated = self.mlpg.evaluate(x=np.array([1, 2, 3])[np.newaxis, :], y=np.array([4, 5, 6, 7])[np.newaxis, :])
        self.assertEqual(self.evaluated.keys(), self.mlpg.parameter_dict.keys())

    def test_key_to_string(self):
        key = (1, 2, 3)
        key_str = MlpGrid.key_to_string(key)
        self.assertEqual(key_str, '1_2_3')
        self.assertEqual(tuple(map(int, key_str.split('_'))), key)

    def test_save_load(self):
        self.mlpg.initialize()
        self.mlpg.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')
        self.mlpg.fit(x=np.random.normal(0, 1, (10, 3)),
                      y=np.random.normal(0, 1, (10, 4)),
                      epochs=1)
        with tempfile.TemporaryDirectory() as dirpath:
            self.mlpg.save(dirpath)
            expected_files = ['MlpGrid_%i_%i.h5' % (i, j) for i in (0, 1) for j in (0, 1, 2)]
            expected_files += ['MlpGrid_history.pickle', 'MlpGrid_parameters.pickle']
            self.assertEqual(sorted(list(os.walk(dirpath))[0][2]), sorted(expected_files))
            self.mlpg_loaded = MlpGrid.from_disk(dirpath)
            self.assertEqual(self.mlpg.parameter_dict.keys(), self.mlpg_loaded.parameter_dict.keys())
            self.assertEqual(self.mlpg.models.keys(), self.mlpg_loaded.models.keys())
            self.assertEqual(self.mlpg.history.keys(), self.mlpg_loaded.history.keys())

    def test_from_param_list(self):
        self.mlp_grid = MlpGrid.from_param_list([np.array([1, 2]), np.array([10, 20, 30])],{'a': 1, 'b': 2})
        self.assertEqual([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                         sorted(list(self.mlp_grid.parameter_dict.keys())))
        np.testing.assert_array_equal(np.array([[1, 10], [1, 20], [1, 30], [2, 10], [2, 20], [2, 30]]),
                                      np.array(list(self.mlp_grid.parameter_dict.values())))
