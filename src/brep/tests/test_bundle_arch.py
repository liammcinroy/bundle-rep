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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Dense, Input
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
            epochs=1000, batch_size=10)

        X = np.array([.1 * x for x in range(0, 10)]).reshape(-1, 1)
        y = np.array([(.1 * x) ** 2 for x in range(0, 10)])

        tf.random.set_seed(0)
        model.fit(X, y, epochs=1000, batch_size=10, verbose=0)

        tf.random.set_seed(0)
        test_model.fit(X, y)

        self.assertTrue(np.sum(np.abs(model.predict_on_batch(X) -
                                      test_model.predict_proba(X))) < 1e-1,
                        'Sequential wrapped Keras models did not perform '
                        'similarly to non-wrapped. Differed by {}'
                        .format(np.sum(np.abs(model.predict_on_batch(X) -
                                              test_model.predict_proba(X)))))

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
                                    epochs=1000, batch_size=10)

        model = TestKerasEstimator.build_simple(Sequential())

        X = np.array([.1 * x for x in range(0, 10)]).reshape(-1, 1)
        y = np.array([(.1 * x) ** 2 for x in range(0, 10)])

        tf.random.set_seed(0)
        model.fit(X, y, epochs=1000, batch_size=10, verbose=0)

        tf.random.set_seed(0)
        test_model.fit(X, y)

        self.assertTrue(np.sum(np.abs(model.predict_on_batch(X) -
                                      test_model.predict_proba(X))) < 1e-1,
                        'Functional wrapped Keras models did not perform '
                        'similarly to non-wrapped. Differed by {}'
                        .format(np.sum(np.abs(model.predict_on_batch(X) -
                                              test_model.predict_proba(X)))))


class TestBRepPlan2VecEstimator(unittest.TestCase):
    """A simple class affirming that the BRepPlan2Vec uses the correct losses
    and parameter sharing. Note that this class does not test the performance
    of bundle representations, that is evaluated elsewhere.
    """

    def simple_rep_dist(self, x1, x2):
        """The desired representation distance for test_simple (l2)

        Arguments:
            x1: a tf tensor
            x2: a tf tensor of the same shape as x1
        """
        return tf.math.sqrt(
            tf.math.reduce_sum(tf.math.square(x1[:, 0] - x2[:, 0]), axis=1))

    def simple_input_dist(self, x1, x2):
        """The desired input distance for test_simple (l2)

        Arguments:
            x1: a tf tensor
            x2: a tf tensor of the same shape as x1
        """
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x1 - x2),
                                               axis=1))

    def test_simple(self):
        """Tests the simplest rep/fiber/reconstruction model available to
        make sure that the weights are shared properly.
        """
        inp = Input(shape=(2,))
        rep = Dense(1, activation='relu')(inp)  # want first
        fiber = Dense(1, activation='relu')(inp)  # want second
        rep_inp = Input(shape=(1,))
        fiber_inp = Input(shape=(1,))
        con = Concatenate()([rep_inp, fiber_inp])
        hid2 = Dense(2, activation='relu')(con)  # want identity
        outp = Dense(2, activation='relu')(hid2)

        rep_model = tf.keras.Model(inputs=inp, outputs=rep)
        fiber_model = tf.keras.Model(inputs=inp, outputs=fiber)
        reconstr_model = tf.keras.Model(inputs=[rep_inp, fiber_inp],
                                        outputs=[outp])
        reconstr_model.compile(loss=keras.losses.mean_squared_error,
                               optimizer=RMSprop())

        model = BRepPlan2VecEstimator(rep_model=rep_model,
                                      fiber_model=fiber_model,
                                      reconstr_model=reconstr_model,
                                      rep_dist=self.simple_rep_dist,
                                      input_dist=self.simple_input_dist,
                                      loss_w=1, optimizer=RMSprop(),
                                      epochs=1, batch_size=100)

        X = np.array([[x1, x2]
                      for x1 in np.linspace(0, 1, 10)
                      for x2 in np.linspace(0, 1, 10)]).reshape(100, 2)

        tf.random.set_seed(0)
        model.fit(X)

        # make sure that the training transferred to the .test_model
        self.assertTrue(np.sum(np.abs(model.predict(X)[0] -
                                      reconstr_model
                                      .predict_on_batch(
                                          [rep_model
                                           .predict_on_batch(X),
                                           fiber_model
                                           .predict_on_batch(X)])[0])) > 1e-5,
                        'BRepPlan2VecEstimator didn\'t share .test_model and '
                        '.train_model weights properly.')

    def test_product_bundle(self):
        """Evaluates plan2vec on a product bundle representation to see if it
        can uncover the two products.
        """
        # we'll have a euclidean metrics, as in the shared test example.
        # So rep = x1, fiber = f(x2) such that f has an inverse.
        X = np.array([[x1, x2]
                      for x1 in np.linspace(0, 1, 10)
                      for x2 in np.linspace(0, 1, 10)]).reshape(100, 2)

        inp = Input(shape=(2,))
        rep = Dense(1, activation='relu')(inp)  # want first
        fiber = Dense(1, activation='relu')(inp)  # want second
        rep_inp = Input(shape=(1,))
        fiber_inp = Input(shape=(1,))
        con = Concatenate()([rep_inp, fiber_inp])
        hid2 = Dense(2, activation='relu')(con)  # want identity
        outp = Dense(2, activation='relu')(hid2)

        rep_model = tf.keras.Model(inputs=inp, outputs=rep)
        fiber_model = tf.keras.Model(inputs=inp, outputs=fiber)
        reconstr_model = tf.keras.Model(inputs=[rep_inp, fiber_inp],
                                        outputs=[outp])
        reconstr_model.compile(loss=keras.losses.mean_squared_error,
                               optimizer=RMSprop())

        model = BRepPlan2VecEstimator(rep_model=rep_model,
                                      fiber_model=fiber_model,
                                      reconstr_model=reconstr_model,
                                      rep_dist=self.simple_rep_dist,
                                      input_dist=self.simple_input_dist,
                                      loss_w=1, optimizer=RMSprop(),
                                      epochs=100, batch_size=100)

        # With the given setup, then the trained model should be able to
        # reconstruct the input and rep = inp[0] and fiber = inp[1]
        tf.random.set_seed(0)
        model.fit(X, verbose=1)


if __name__ == '__main__':
    unittest.main()
