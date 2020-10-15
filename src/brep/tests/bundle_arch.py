# brep/tests/bundle_arch.py
#
# This file outlines the architecture of a non-RL fiber bundle representation
# These motivate fiber bundle representations, which are used in RL
# environments in other brep/tests files.
#
# Developed by Liam McInroy, 2020/6/23.


import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input


class KerasEstimator(BaseEstimator, RegressorMixin):
    """An sklearn wrapper for a keras network.
    """

    def __init__(self, model=None, epochs=None, batch_size=None):
        """Initializes a new arbitrary keras netowrk and uses it. The given
        model should already have been compiled.

        Arguments:
            model: The keras.models.Sequential to be used. Make sure it has
                already been compiled before calling __init__.
            epochs: The number of epochs to train on.
            batch_size: The batch size to use during training.
        """
        self.model = clone_model(model)
        self.model.set_weights(model.get_weights())
        self.model.compile(loss=model.loss, optimizer=model.optimizer,
                           metrics=model.metrics,
                           loss_weights=model.loss_weights,
                           sample_weight_mode=model.sample_weight_mode)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fits the keras model to X, y

        Arguments:
            X: The features of the dataset
            y: The labels of the dataset.
        """
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=0)

        return self

    def predict(self, X, y=None):
        """Predicts using the keras model on X
        Arguments:
            X: The features of the set to predict on
        """
        return self.model.predict_on_batch(X)

    def predict_proba(self, X, y=None):
        """Predicts the probabilities using the keras model on X.
        When using a generator, then sampling is done (since this is
        a classifier) so this gives the raw probabilities
        Arguments:
            X: The features of the dataset to get probability classes for
        """
        return self.model.predict_on_batch(X)

    def score(self, X, y=None):
        """Scores the model using keras
        Arguments:
            X: The features of the set to test on
            y: The corresponding labels
        """
        return self.model.evaluate(X, y)

    def loss(self, X, y=None):
        """Gets the loss metric
        Arguments:
            X: The features of the set to test on
            y: The corresponding labels
        """
        return self.model.evaluate(X, y)


class BRepPlan2VecKerasTrainModel(keras.Model):
    """A class implementing a simple bundle representation network with the
    Plan2Vec loss. Overrides the behavior of a normal keras.Model for the
    customized losses.
    """

    def __brep_plan2vec_loss__(self, reconstr1, rep1, reconstr2, rep2):
        """Defines the custom Bundle Representation Plan2Vec loss. For valid
        pairs, then the Bundle Representantion Plan2Vec loss is a regression
        of the distance along reconstructions to match the distance of the base
        manifold of the representations, since reconstr2 is reconstructed with
        the task-irrelevant fiber of rep1
        """
        dist_reconstr = self.input_dist(reconstr1, reconstr2)
        dist_rep = self.rep_dist(rep1, rep2)

        # TODO use a non-MSE for non-euclidean manifolds
        return self.loss_w * keras.losses.MSE(dist_reconstr, dist_rep)

    def __init__(self, inputs=None, outputs=None, reconstr_loss=None,
                 rep_dist=None, input_dist=None, loss_w=None,
                 optimizer=None, reconstr_model_name=None):
        """Initializes a new test keras network for use with BRepPlan2Vec.
        Will create a deep copy of the given networks to construct its own.
        If the models have additional losses, then those will be used during
        training as well. Don't call compile on this model!

        Arguments:
            inputs: The tf input tensors used for the model. Should be in1, in2
            outputs: The tf output tensors used for the model. Should be
                [rep1, fiber1, reconstr1, rep2, reconstr2].
            reconstr_loss: A tf loss on the reconstructed output.
            rep_dist: The distance function to use in representation space.
                Should be constructed using tensorflow/keras so that backprop
                is possible. Should accept two tensors (of equal shape).
            input_dist: The distance function to use in representation space.
                Stop gradient will be used anyways, so no need use tensorflow
                or keras functions.
            loss_w: The weight to apply to the Bundle Representation Plan2Vec
                loss.
            optimizer: The tensorflow.keras.optimizers.Optimizer to be used
                in each submodel.
            reconstr_model_name: The name to pass to the loss on reconstr1.
        """
        # Now we can create the train model
        super(BRepPlan2VecKerasTrainModel, self).__init__(inputs=inputs,
                                                          outputs=outputs)

        # We keep the reconstruction loss for use during training.
        self.reconstr_loss = reconstr_loss

        # We keep the representation distance metric and input distance
        # functions (as differentiable tensors for autodifferentiation)
        self.rep_dist = rep_dist
        self.input_dist = input_dist

        # record the training params
        self.loss_w = loss_w
        self.optimizer = optimizer

        # create the loss trackers
        self.reconstr_loss_tracker = keras.metrics.Mean(
            name='reconstruction_loss')

        self.brep_loss_tracker = keras.metrics.Mean(
            name='bundle_representation_loss')

        # configure the optimizer with compile
        self.compile(loss={reconstr_model_name:
                           self.reconstr_loss},
                     optimizer=optimizer)

        def train_step(self, data):
            """The keras train_step used in train. We override to apply the
            Bundle Representation Plan2Vec training loss as well.

            Arguments:
                data: The training data. Should be X of pairs of inputs and
                    y is the first pair item input.
            """
            X, y = data

            with tf.GradientTape() as tape:
                rep1, fiber1, reconstr1, rep2, reconstr2 = self(X,
                                                                training=True)
                reconstr_loss = self.reconstr_loss(y, reconstr1)
                brep_loss = self.compiled_loss(reconstr1, rep1,
                                               reconstr2, rep2)
                loss = {self.reconstr_loss_tracker.name: reconstr_loss,
                        self.brep_loss_tracker.name: brep_loss}

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(self.loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.reconstr_loss_tracker.update_state(
                loss[self.reconstr_loss_tracker.name])
            self.brep_loss_tracker.update_state(
                loss[self.brep_loss_tracker.name])

            return {self.reconstr_loss_tracker.name:
                    self.reconstr_loss_tracker.result(),
                    self.brep_loss_tracker.name:
                    self.brep_loss_tracker.result()}

        @property
        def metrics(self):
            """Automatically call reset_state() at each epoch.
            """
            return [self.reconstr_loss_tracker, self.brep_loss_tracker]


class BRepPlan2VecEstimator(KerasEstimator):
    """A class implementing a simple bundle representation network with the
    Plan2Vec loss. Uses the KerasEstimator class to form an sklearn type.
    """

    def __preprocess_inputs__(self, X, y=None):
        """Preprocesses the input batch into the form required for training,
        loss, scoring by Bundle Representation Plan2Vec.

        Arguments:
            X: The features of the batch training set.
            y: If empty, then the labels are autogenerated. If non-empty, then
                interpreted as the true distance of states (in the maximal
                bundle representation).
        """
        new_X = [(x1, x2)
                 for (i1, x1) in enumerate(X)
                 for (i2, x2) in enumerate(X) if i1 != i2]
        new_y = np.array([pair_x[0] for pair_x in new_X])
        return [np.array([x[0] for x in new_X]).reshape(len(new_X),
                                                        *X[0].shape),
                np.array([x[1] for x in new_X]).reshape(len(new_X),
                                                        *X[0].shape)], new_y

    def __init__(self, rep_model=None, fiber_model=None, reconstr_model=None,
                 reconstr_loss=None, rep_dist=None, input_dist=None,
                 loss_w=None, optimizer=None, epochs=None, batch_size=None):
        """Initializes a new test keras network for use with BRepPlan2Vec.
        Will create a deep copy of the given network. If the model has
        additional losses, then those will be used during training as well.
        Arguments:
            rep_model, fiber_model, reconstr_model: The architecture to be
                used. To ease development of this for arbitrary models, then
                we require the three separate models (each with one output).
                We presume rep_model and fiber_model share many parameters
                (although not necessarily so), but have distinct outputs.
                The rep_model is intended to capture the representation
                of inputs, specifically S_\phi, while fiber_model should
                output the fiber of the bundle, i.e. W_\phi. Then,
                reconstr_model should accept the outputs of both rep_model
                and fiber_model as inputs, which are concatenated, and then
                produce a reconstruction. This allows for some flexibility,
                i.e. a VAE could be used for reconstructions etc.
            reconstr_loss: A tf loss on the reconstructed output.
            rep_dist: The distance function to use in representation space.
                Should be constructed using tensorflow/keras so that backprop
                is possible. Should accept two tensors (of equal shape).
            input_dist: The distance function to use in representation space.
                Stop gradient will be used anyways, so no need use tensorflow
                or keras functions.
            loss_w: The weight to apply to the Bundle Representation Plan2Vec
                loss.
            optimizer: A tf optimizer to apply to the Bundle Representation
                training network.
            epochs: The number of epochs to train on.
            batch_size: The batch size to use during training.
        """
        # Here we construct a net for training and for testing. We'll use
        # shared layers between the two so that they agree. We need two, since
        # the Bundle Representation Plan2Vec loss requires two inputs whereas
        # we only want the representation to predict for single inputs.
        self.rep_model = clone_model(rep_model)
        self.rep_model.set_weights(rep_model.get_weights())
        self.fiber_model = clone_model(fiber_model)
        self.fiber_model.set_weights(fiber_model.get_weights())
        self.reconstr_model = clone_model(reconstr_model)
        self.reconstr_model.set_weights(reconstr_model.get_weights())

        # We keep the reconstruction loss for use during training.
        self.reconstr_loss = reconstr_loss

        # We keep the representation distance metric and input distance
        # functions (as differentiable tensors for autodifferentiation)
        self.rep_dist = rep_dist
        self.input_dist = input_dist

        # record the training params
        self.loss_w = loss_w
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        # We now create the test model.
        in_test = Input(shape=rep_model.inputs[0].shape[1:])
        rep_test = self.rep_model(in_test)
        fiber_test = self.fiber_model(in_test)
        reconstr_test = self.reconstr_model([rep_test, fiber_test])
        self.test_model = keras.Model(inputs=in_test,
                                      outputs=[rep_test, fiber_test,
                                               reconstr_test])

        # Now we create the training model. Two copies for each train pair.
        in1 = Input(shape=rep_model.inputs[0].shape[1:])
        in2 = Input(shape=rep_model.inputs[0].shape[1:])
        # We can just call the test model on the first input.
        rep1 = self.rep_model(in1)
        fiber1 = self.fiber_model(in1)
        reconstr1 = self.reconstr_model([rep1, fiber1])
        # In the second, we need to replace the fiber value of model(in2)
        # with that of model(in1).
        rep2 = self.rep_model(in2)
        reconstr2 = self.reconstr_model([rep2, fiber1])  # new W_\phi value
        # Now we can create the train model
        self.train_model = BRepPlan2VecKerasTrainModel(
            inputs=[in1, in2], outputs=[rep1, fiber1, reconstr1,
                                        rep2, reconstr2],
            reconstr_loss=self.reconstr_loss, rep_dist=self.rep_dist,
            input_dist=self.input_dist, loss_w=self.loss_w,
            optimizer=self.optimizer,
            reconstr_model_name=self.reconstr_model.name)

    def fit(self, X, y=None):
        """Fits the given model to the given features and labels (which should
        just be the original features). In other words, the model should be
        capable firstly of reconstructing states. Additionally, there should
        be an auxiliary loss on the distance in the latent space (specifically
        that of the base maximal bundle representation) such that lifts along
        the fiber have distance equal in the feature space to that of the
        latent representation. This auxiliary loss is expressed in the hidden
        __brep_plan2vec_loss__ function.

        Note that the training method will use a batch of observation X that
        is processed by this method into a training set including the proper
        labels

        Arguments:
            X: The features of the batch training set.
            y: Should be empty. Will be ignored regardless.
        """
        self.train_model.fit(*self.__preprocess_inputs__(X),
                             epochs=self.epochs, batch_size=self.batch_size,
                             verbose=0)
        return self

    def predict(self, X):
        """Gives the predicted rep, fiber, reconstruction outputs for the
        given input using self.test_model, which shares weights with
        self.train_model.

        Arguments:
            X: The features of the set to predict on.
        """
        return self.test_model.predict_on_batch(X)

    def score(self, X, y=None):
        """Scores the model using keras. Again used the preprocessing step
        discussed by fit(...).

        Arguments:
            X: The features of the set to test on
            y: The corresponding labels. If empty, then approximates. Else,
                then the labels are intrepreted as the true distance.
        """
        return self.train_model.evaluate(*self.__preprocess_inputs__(X))

    def loss(self, X, y=None):
        """Gets the loss metric. Again used the preprocessing step
        discussed by fit(...).

        Arguments:
            X: The features of the set to test on
            y: Should b eempty. Will be ignored regardless.
        """
        return self.train_model.evaluate(*self.__preprocess_inputs__(X))
