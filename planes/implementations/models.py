import numpy as np
import tensorflow as tf

from gpflow.base import MeanAndVariance, InputData
from gpflow.models import SVGP
from gpflow import posteriors

class SVGP_with_min_var(SVGP):
    """
    Clip variance to a small positive constant from below to prevent negative variances.
    Until now issue was encountered only when kernel was a sum kernel with a lot of components (e.g. convolutional structure).
    Maybe we should include an option to specify the constant explicitly.
    """
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        fmean, fvar = self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return fmean, tf.maximum(fvar, 1e-5)