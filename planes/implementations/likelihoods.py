import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Callable, Optional, Type

import gpflow

from gpflow.base import TensorType, MeanAndVariance
from gpflow import logdensities

from gpflow.likelihoods.multiclass import Softmax
from gpflow.likelihoods.multilatent import MultiLatentLikelihood
from gpflow.likelihoods.utils import inv_probit


class SoftmaxFixed(Softmax):
    '''
    Modification to method _log_prob required.
    See: https://github.com/GPflow/GPflow/issues/1591
    '''
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=tf.argmax(Y, axis=1))