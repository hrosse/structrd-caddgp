from typing import Union

import numpy as np 
import tensorflow as tf

from gpflow.base import TensorLike, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
import gpflow.covariances as covs
from gpflow.inducing_variables import InducingPatches

from.kernels import ConvolutionalBase


@covs.Kuf.register(InducingPatches, ConvolutionalBase, object)
def Kuf_convbase_patch(
    inducing_variable: InducingPatches, kernel: ConvolutionalBase, Xnew: TensorType
) -> tf.Tensor:
    Xp = kernel.get_patches(Xnew)
    bigKzx = kernel.base_kernel.K(
        inducing_variable.Z, Xp
    ) 
    Kzx = tf.reduce_sum(bigKzx * kernel.get_weights, [2])
    return Kzx  / (to_default_float(kernel.num_patches) / kernel.normalisation_factor_scale)



@covs.Kuu.register(InducingPatches, ConvolutionalBase)
def Kuu_convbase_patch(
    inducing_variable: InducingPatches, kernel: ConvolutionalBase, jitter: float = 0.0
) -> tf.Tensor:
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        inducing_variable.num_inducing, dtype=default_float()
    )
