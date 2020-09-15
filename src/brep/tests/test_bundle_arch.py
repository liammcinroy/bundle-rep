# brep/tests/test_bundle_arch.py
#
# This file is used to test the bundle representation architecture on non-RL
# tasks. Specifically, we use construct a trivial product bundle and a twisted
# bundle to test whether the architecture is capable of extracting the bundle
# structure. These motivate fiber bundle representations, which are used in RL
# environments in other brep/tests files.
#
# Developed by Liam McInroy, 2020/9/11.


import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop

from bundle_arch import KerasEstimator, BRepPlan2VecEstimator


class TestKerasEstimator(unittest.TestCase):
    """A simple class affirming that the KerasEstimator agrees with an
    unwrapped Keras estimator.
    """

    def build_simple(model):
        """Adds simple dense layers, compiles the model.

        Arguments:
            model: A Keras Sequential model
        """
        model.add(Dense(10, activation='relu', input_dim=1))
        model.add(Dense(1))

        rmsprop = RMSprop()
        model.compile(loss='mean_squared_error',
                      optimizer=rmsprop)

        return model

    def test_sequential(self):
        """Tests that a "simple" Sequential keras model and one wrapped by
        KerasEstimator perform equally (and implicitly that cloning works).
        """
        model = TestKerasEstimator.build_simple(Sequential())

        test_model = KerasEstimator(
            TestKerasEstimator.build_simple(Sequential()),
            epochs=500, batch_size=10)

        X = np.array([.1 * x for x in range(0, 10)]).reshape(-1, 1)
        y = np.array([(.1 * x) ** 2 for x in range(0, 10)])

        model.fit(X, y, epochs=500, batch_size=10, verbose=0)
        test_model.fit(X, y)

        self.assertTrue(abs(np.sum(model.predict_on_batch(X) -
                                   test_model.predict(X))) < 1e-1)

    def test_models(self):
        """Tests a non-sequential Keras model, and one wrapped in
        KerasEstimator, to affirm they perform equally.
        """
        inp = Input(shape=(1,))
        hidden = Dense(10, activation='relu')(inp)
        out = Dense(1)(hidden)
        keras_model = tf.keras.Model(inputs=inp, outputs=out)
        keras_model.compile(loss='mean_squared_error',
                            optimizer=RMSprop())
        test_model = KerasEstimator(keras_model,
                                    epochs=500, batch_size=10)

        model = TestKerasEstimator.build_simple(Sequential())

        X = np.array([.1 * x for x in range(0, 10)]).reshape(-1, 1)
        y = np.array([(.1 * x) ** 2 for x in range(0, 10)])

        model.fit(X, y, epochs=500, batch_size=10, verbose=0)
        test_model.fit(X, y)

        self.assertTrue(abs(np.sum(model.predict_on_batch(X) -
                                   test_model.predict(X))) < 1e-1)


class TestBRepPlan2VecEstimator(unittest.TestCase):
    """A simple class affirming that the BRepPlan2Vec uses the correct losses
    and parameter sharing. Note that this class does not test the performance
    of bundle representations, that is evaluated elsewhere.
    """

    def test_simple(self):
        """Tests the simplest rep/fiber/reconstruction model available to
        make sure that the wrapped class performs equally to non-wrapped.
        """
        raise NotImplementedError()

    def test_shared(self):
        """Tests that rep/fiber/reconstruction model with shared weights
        and the wrapped class performs equally to non-wrapped.
        """
        raise NotImplementedError()

    def test_aux_loss(self):
        """Tests the rep/fiber/reconstruction model with a predefined loss
        and its the wrapped class performs equally to non-wrapped.
        """
        raise NotImplementedError()


if __name__ == '__main__':
    unittest.main()
