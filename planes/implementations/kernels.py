import abc

from typing import Optional, Sequence, cast, List 

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel



class ACRDKernel(Kernel):
    '''
    Implements automatic channel relevance detection (ACRD).
    Dimensions belonging to the same channel are scaled by the same lengthscale.
    
    The channel-additive kernel is constructed as Sum kernel where each base kernel is an ACRDKernel.
    The attribute active_channels is used to  specify the channels of each base kernel.

    If active_channels is specified the argument is expected to be a list of channel indices.
    E.g. active_channels=[0,1] specifies that only the first two image channels are used in the kernel.
    The inputs are sliced before handing the sliced inputs to kernel_instance.
    '''
    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        colour_channels: int,
        active_channels: Optional[Sequence[int]] = None,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.active_channels = active_channels
        self.n_active = self.colour_channels if self.active_channels is None else len(self.active_channels)
        self.lengthscales = Parameter(
            np.ones(self.n_active, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def slice_channels(self, X:TensorType) -> TensorType:
        '''
        X must have shape like [..., C]
        '''
        return tf.gather(params=X, indices=self.active_channels, axis=-1)

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, L] where L is dimensionality of data after slicing
        '''
        if X is not None:
            input_shape = tf.shape(X)
            output_shape = tf.concat([input_shape[:-1], [self.n_active * self.image_shape[0] * self.image_shape[1]]], 0)
            bNxWxH = tf.reduce_prod(input_shape[:-1]) * self.image_shape[0] * self.image_shape[1]
            X_scaled = tf.reshape(X, [bNxWxH, self.colour_channels]) # shape [num_data_x_W_x_H, C]
            # Slice input when active channels are specified
            if self.active_channels is not None:
                X_scaled = self.slice_channels(X_scaled) # shape [num_data_x_W_x_H, L]
            X_scaled = X_scaled / self.lengthscales # usually sufficient
            X_scaled = tf.reshape(X_scaled, output_shape) # shape [batch..., N, L]
        else:
            X_scaled = X
        return X_scaled

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.kernel_instance.K_diag(self.scale(X))



class APRDKernel(Kernel):
    '''
    Implements automatic patch relevance detection (APRD).
    Dimensions belonging to the same patch are scaled by the same lengthscale.
    APRD uses the same lengthscale for all channels in a patch.
    
    For a plain APRD kernel an isotropic RBF kernel can be used as kernel_instance.
    Set lengthscales of kernel_instance to non-trainable to avoid parameter redundancy.
    Hyperparameter variance of kernel_instance should be trainable because APRDKernel
    does not have this parameter.
    
    For the combination of APRD with ACRD a ACRD kernel can be used as kernel_instance.
    
    For combination of APRD and channel-additive kernels a channel-additive kernel is used as kernel_instance.
    '''
    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        patch_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.lengthscales = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, D]
        '''
        if X is not None:
            input_shape = tf.shape(X)
            batch = tf.shape(X)[:-2]
            N = tf.shape(X)[-2]
            flat_batch = tf.reduce_prod(batch)
            num_data = flat_batch * N
            X = tf.reshape(
                X,
                [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
            ) # shape [num_data, W, H, C]

            # if argument sizes and strides are equal extract_patches extracts disjoint patches
            Xp = tf.image.extract_patches(
                X,
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, 1, 1, 1],
                "VALID",
            ) # shape [num_data, n_x_patches, n_y_patches, w_x_h_x_C]
            shp = tf.shape(Xp)

            lengthscales_reshaped = tf.reshape(
                self.lengthscales, [1, shp[1], shp[2], 1]
            ) # shape [1, n_x_patches, n_y_patches, 1]
            lengthscales_reshaped = tf.repeat(
                lengthscales_reshaped, repeats=self.len_patches, axis=-1
            ) # shape [1, n_x_patches, n_y_patches, C_x_w_x_h]
            
            # Included to_default_float(Xp) just in case.
            X_scaled = to_default_float(Xp) / to_default_float(lengthscales_reshaped)

            X_scaled = tf.reshape(
                X_scaled,
                [num_data, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
            ) # shape [num_data, n_x_patches, n_y_patches, w, h, C]

            X_scaled = tf.transpose(
                X_scaled, [0, 1, 3, 2, 4, 5]
            )# shape [num_data, n_x_patches, w, n_y_patches, h, C]

            X_scaled = tf.reshape(X_scaled, input_shape)
            X_scaled = to_default_float(X_scaled)
        else:
            X_scaled = X
        return X_scaled

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.kernel_instance.K_diag(self.scale(X))

    @property
    def num_patches(self) -> int:
        '''
        number of channels is irrelevant.
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    self.image_shape[0] / self.patch_shape[0]
                ) * (
                    self.image_shape[1] / self.patch_shape[1]
                )
            ),
            dtype=tf.int32
        )

    @property
    def len_patches(self) -> int:
        return tf.cast(
            tf.reduce_prod(
                self.patch_shape[0] * self.patch_shape[1] * self.colour_channels
            ),
            dtype=tf.int32
        )



class PCAPRDKernel(Kernel):
    '''
    Implements a patch-channel APRD.
    Dimensions belonging to the same patch are scaled by the same lengthscale.
    Partitions each channel into patches.
    '''
    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        patch_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.lengthscales = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, D]
        '''
        if X is not None:
            input_shape = tf.shape(X)
            batch = tf.shape(X)[:-2]
            N = tf.shape(X)[-2]
            flat_batch = tf.reduce_prod(batch)
            num_data = flat_batch * N
            X = tf.reshape(
                X,
                [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
            ) # shape [num_data, W, H, 3]
            # if argument sizes and strides are equal extract_patches extracts disjoint patches
            Xp = tf.image.extract_patches(
                X,
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, 1, 1, 1],
                "VALID",
            ) # shape [num_data, n_x_patches, n_y_patches, w_x_h_x_C]
            shp = tf.shape(Xp)
            Xp = tf.reshape(
                Xp,
                [num_data, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
            ) # shape [num_data, n_x_patches, n_y_patches, w, h, C]
            Xp = tf.transpose(
                Xp, [0, 5, 1, 2, 3, 4]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w, h]
            Xp = tf.reshape(
                Xp,
                [num_data, self.colour_channels, shp[1], shp[2], tf.reduce_prod(self.patch_shape)]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w_x_h]

            lengthscales_reshaped = tf.reshape(
                self.lengthscales, [1, self.colour_channels, shp[1], shp[2], 1]
            ) # shape [1, C, n_x_patches, n_y_patches, 1]
            lengthscales_reshaped = tf.repeat(
                lengthscales_reshaped, repeats=tf.reduce_prod(self.patch_shape), axis=-1
            ) # shape [1, C, n_x_patches, n_y_patches, w_x_h]

            # Included to_default_float(Xp) just in case.
            X_scaled = to_default_float(Xp) / to_default_float(lengthscales_reshaped)

            # transpose back so that input is in same order as output
            # relevant in case we have kernel instance that is combination of
            # kernels with active dimensions specified.
            X_scaled = tf.reshape(
                X_scaled,
                [num_data, self.colour_channels, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1]]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w, h]
            X_scaled = tf.transpose(
                X_scaled, [0, 2, 3, 4, 5, 1]
            )# shape [num_data, n_x_patches, n_y_patches, w, h, C]
            
            # here additionally transposing X_scaled to shape [num_data, n_x_patches, w, n_y_patches, h, C]
            X_scaled = tf.transpose(
                X_scaled, [0, 1, 3, 2, 4, 5]
            )# shape [num_data, n_x_patches, w, n_y_patches, h, C]
            X_scaled = tf.reshape(X_scaled, input_shape)
            X_scaled = to_default_float(X_scaled)
        else:
            X_scaled = X
        return X_scaled

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.kernel_instance.K_diag(self.scale(X))

    @property
    def num_patches(self) -> int:
        return tf.cast(
            tf.reduce_prod(
                (
                    self.image_shape[0] / self.patch_shape[0]
                ) * (
                    self.image_shape[1] / self.patch_shape[1]
                ) * self.colour_channels
            ),
            dtype=tf.int32
        )
        
        
        
        
class ConvolutionalBase(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        normalisation_factor_scale: int = 1
    ) -> None:
        super().__init__()
        self.base_kernel = base_kernel
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.colour_channels = colour_channels
        self.strides = strides
        self.normalisation_factor_scale = to_default_float(normalisation_factor_scale)
    
    @abc.abstractmethod
    def get_patches(self, X: TensorType) -> tf.Tensor:
        raise NotImplementedError
        
    @property
    @abc.abstractmethod
    def get_weights(self) -> TensorType:
        raise NotImplementedError
        
    @property
    @abc.abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError
        
        
        

class FullConvolutional(ConvolutionalBase):
    r"""
    The implementation in gpflow 2.9.2 extracts patches from each colour channel.
    This kernel is a slight modification of the convolutional kernel where each patch contains the values of all channels.
    Additionally the strides used in tf.image.extract_patches can be specified explicitly.
    For SVGP with inter-domain inducing variables use class InducingPatches.
    Instead of accessing the weights directly, they are accessed via get_weights.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        weights: Optional[TensorType] = None,
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        normalisation_factor_scale: int = 1
    ) -> None:
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            colour_channels=colour_channels,
            strides=strides,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        self.weights = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if weights is None else weights
        )
        
        
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        Do not roll channel in front but hand multi-channel images as input to tf.image.extract_patches. 
        """
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(
            X, 
            [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
        ) # "[num_data, W, H, C]"
        patches = tf.image.extract_patches(
            X,
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, self.strides[0], self.strides[1], 1],
            [1, 1, 1, 1],
            "VALID",
        ) #"[num_data, n_x_patches, n_y_patches, S]" where S=w_x_h_x_C
        shp = tf.shape(patches)
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2], shp[3]]], 0)
        ) # "[batch..., N, P, S]"
        return to_default_float(reshaped_patches)
    
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.weights[:, None] * self.weights[None, :] # "[P, P]"

        rank = tf.rank(Xp) - 3
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        P = tf.shape(Xp)[-2]
        S = tf.shape(Xp)[-1]
        ones = tf.ones((rank,), dtype=tf.int32)

        if X2 is None:
            Xp = tf.reshape(Xp, tf.concat([batch, [N * P, S]], 0)) # "[batch..., N_x_P, S]"
            bigK = self.base_kernel.K(Xp) # "[batch..., N_x_P, N_x_P]"
            bigK = tf.reshape(
                bigK, tf.concat([batch, [N, P, N, P]], 0)
            ) # "[batch..., N, P, N, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P, 1, P]], 0)
            ) # "[..., 1, P, 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, N, P]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0 # "[batch..., N, N]"
        else:
            Xp2 = Xp if X2 is None else self.get_patches(X2) # "[batch2..., N2, P, S]"
            rank2 = tf.rank(Xp2) - 3
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            bigK = self.base_kernel.K(Xp, Xp2) # "[batch..., N, P, batch2..., N2, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P], ones2, [1, P]], 0)
            ) #"[..., 1, P, ..., 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, batch2..., N2, P]"
            # shape of returned final tensor is "[batch..., N, batch2..., N2]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0
    
    def K_diag(self, X: TensorType) -> tf.Tensor:
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        rank = tf.rank(Xp) - 3
        P = tf.shape(Xp)[-2]
        ones = tf.ones((rank,), dtype=tf.int32)
        W2 = self.weights[:, None] * self.weights[None, :] # "[P, P]"
        W2 = tf.reshape(W2, tf.concat([ones, [1, P, P]], 0)) # "[..., 1, P, P]"
        bigK = self.base_kernel.K(Xp) # "[batch..., N, P, P]"
        return tf.reduce_sum(bigK * W2, [rank + 1, rank + 2]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0
    
    @property
    def get_weights(self) -> TensorType:
        return self.weights
    
    @property
    def num_patches(self) -> int:
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                )
            ),
            dtype=tf.int32
        )
