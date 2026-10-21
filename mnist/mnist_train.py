import os
import sys
import json
import time
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import default_float
from gpflow.utilities import to_default_float, to_default_int

from implementations.models import SVGP_with_min_var

from gpflow.kernels import SquaredExponential, SeparateIndependent
from implementations.kernels import APRDKernel

from gpflow.inducing_variables import InducingPoints, SeparateIndependentInducingVariables, FallbackSeparateIndependentInducingVariables

from implementations.likelihoods import SoftmaxFixed
from gpflow.likelihoods import MultiClass, RobustMax, Gaussian

# Disable too strict shape checking.
# Inducing points can have different dimensionality.
FallbackSeparateIndependentInducingVariables.__init__ = FallbackSeparateIndependentInducingVariables.__init__.__wrapped__



def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPflow SVGP model")

    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    parser.add_argument("--rseed", type=int, default=101,
                        help="Random seed for reproducibility")

    # Directory paths and number of points included in training data
    parser.add_argument("--data_dir", type=str, default="./data/mnist",
                        help="Directory containing dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/mnist",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--n_train", type=int, default=-1,
                        help="Number of points sampled from training data, default -1 for full training data")

    # Default float options and default jitter
    parser.add_argument("--default_dtype", type=str, choices=["f32", "f64", "f16"], default="f32",
                        help="Default dtype (maps to np.float32, np.float64, np.float16)")
    parser.add_argument("--default_jitter", type=float, default=1e-5,
                        help="Default jitter value")

    # Model options, inducing points number and sampling options
    parser.add_argument("--model", type=str,
                        choices=["isotropic", "ARD", "APRD2x2", "APRD4x4", "APRD7x7"],
                        required=True,
                        help="Model type")
    parser.add_argument("--M", type=int, required=True,
                        help="Number of inducing points")
    parser.add_argument("--im_sampling", type=str, choices=["data", "uniform", "mixed"], default="data",
                        help="Sample strategy for inducing images")
    parser.add_argument("--likelihood", type=str, choices=["softmax", "robustmax", "gaussian"], default="softmax",
                        help="Likelihood for the model")

    ### Optimisation options

    parser.add_argument("--mb_size", type=int, required=True,
                        help="Minibatch size")

    # Stopping Conditions for training
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum number of training steps")
    parser.add_argument("--max_time", type=float, default=None,
                        help="Maximum training time in seconds")

    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Learning rate and learning rate schedule
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                        help="Steps after which to reduce learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=None,
                        help="Factor with which to multiply learning rate")

    args = parser.parse_args()

    # Map dtype string to numpy dtype
    dtype_map = {"f32": np.float32, "f64": np.float64, "f16": np.float16}
    args.default_dtype = dtype_map[args.default_dtype]

    args = vars(args)

    # Separate directory for each selected model in specified checkpoint directory
    args["save_dir"] = args["save_dir"] + "/" + args["model"]

    return args


def load_data(arg_dict):
    """
    Returns (Xtrain, Ytrain, Xtest, Ytest, arg_dict).
    """
    # Load full dataset first
    xtrain_path = os.path.join(arg_dict["data_dir"], "xtrain.npz")
    if not os.path.exists(xtrain_path):
        raise FileNotFoundError(f"No dataset found at {xtrain_path}")
    ytrain_path = os.path.join(arg_dict["data_dir"], "ytrain.npz")
    if not os.path.exists(ytrain_path):
        raise FileNotFoundError(f"No dataset found at {ytrain_path}")
    xtest_path = os.path.join(arg_dict["data_dir"], "xtest.npz")
    if not os.path.exists(xtest_path):
        raise FileNotFoundError(f"No dataset found at {xtest_path}")
    ytest_path = os.path.join(arg_dict["data_dir"], "ytest.npz")
    if not os.path.exists(ytest_path):
        raise FileNotFoundError(f"No dataset found at {ytest_path}")

    with np.load(xtrain_path) as data:
        Xtrain = data["arr_0"]
    with np.load(ytrain_path) as data:
        Ytrain = data["arr_0"]
    with np.load(xtest_path) as data:
        Xtest = data["arr_0"]
    with np.load(ytest_path) as data:
        Ytest = data["arr_0"]
        
    # If n_train specified as command line argument sample n_train points
    if arg_dict["n_train"] > 0:
        mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["n_train"], replace=False)
        Xtrain = Xtrain[mask]
        Ytrain = Ytrain[mask]
     
    num_data = Xtrain.shape[0] # Number of points in training data.
    image_shape = Xtrain.shape[1:-1]
    colour_channels = Xtrain.shape[-1]
    num_classes = np.unique(Ytrain).shape[0]

    # Flatten images
    Xtrain = np.reshape(Xtrain, [Xtrain.shape[0], np.prod(Xtrain.shape[1:])])
    Xtest = np.reshape(Xtest, [Xtest.shape[0], np.prod(Xtest.shape[1:])])

    # Required arguments when building models, depend on the data
    arg_dict["num_data"] = num_data
    arg_dict["num_classes"] = num_classes
    arg_dict["image_shape"] = image_shape
    arg_dict["colour_channels"] = colour_channels

    # Encoding of labels as required by likelihood
    if arg_dict["likelihood"] == "softmax":
        Ytrain_hot = np.zeros((num_data, num_classes), dtype=int)
        Ytrain_hot[np.arange(num_data), Ytrain.astype("int")] = 1.0
        Ytrain = to_default_int(Ytrain_hot)
        Ytest_hot = np.zeros((Xtest.shape[0], num_classes), dtype=int)
        Ytest_hot[np.arange(Xtest.shape[0]), Ytest.astype("int")] = 1.0
        Ytest = to_default_int(Ytest_hot)
    elif arg_dict["likelihood"] == "robustmax":
        Ytrain = to_default_int(Ytrain)
        Ytest = to_default_int(Ytest)
    elif arg_dict["likelihood"] == "gaussian":
        Ytrain_hot = np.zeros((num_data, num_classes), dtype=int)
        Ytrain_hot[np.arange(num_data), Ytrain.astype("int")] = 1.0
        Ytrain = to_default_float(Ytrain_hot)
        Ytest_hot = np.zeros((Xtest.shape[0], num_classes), dtype=int)
        Ytest_hot[np.arange(Xtest.shape[0]), Ytest.astype("int")] = 1.0
        Ytest = to_default_float(Ytest_hot)
        
    # Cast data to default float, returns tensors
    Xtrain = to_default_float(Xtrain)
    Xtest = to_default_float(Xtest)

    return Xtrain, Ytrain, Xtest, Ytest, arg_dict


def build_model(Xtrain, arg_dict):
    # Block below is used as standard values in experiments.
    # Values might be replaced in the code blocks where specific models are build. 
    var_init_image = 1.0
    lscale_init_image = 5.0
    patch_shape_aprd = [2, 2]

    if arg_dict["likelihood"] == "softmax":
        likelihood = SoftmaxFixed(arg_dict["num_classes"])
    elif arg_dict["likelihood"] == "robustmax":
        invlink = RobustMax(arg_dict["num_classes"])
        likelihood = MultiClass(arg_dict["num_classes"], invlink=invlink)
    elif arg_dict["likelihood"] == "gaussian":
        likelihood = Gaussian()

    # Different sample strategies for inducing images.
    def sample_inducing_images(Xtrain, arg_dict, kernel=None, r_uniform=0.5):
        if arg_dict["im_sampling"] == "data":
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z = Xtrain.numpy()[mask]
        elif arg_dict["im_sampling"] == "uniform":
            Z = np.random.randint(0, 255, size=(arg_dict["M"], Xtrain.shape[-1]))/255
        elif arg_dict["im_sampling"] == "mixed":
            Z_uniform = np.random.randint(0, 255, size=(arg_dict["M"], Xtrain.shape[-1]))/255
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z_data = Xtrain.numpy()[mask]
            Z = r_uniform * Z_uniform + (1 - r_uniform) * Z_data
        return InducingPoints(Z)

    def initialise_SVGP_model(kernel, likelihood, Z, arg_dict):
        model = SVGP_with_min_var(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=Z,
            num_data=arg_dict["num_data"],
            num_latent_gps=arg_dict["num_latent"]
        )
        return model

    # Below are some convenient functions that return bijectors tailored to specific models/kernels.
    # Define constraints on kernel hyperparameters.
    def affine_scalar_bijector(shift=None, scale=None):
        scale_bijector = (
            tfp.bijectors.Scale(scale) if scale else tfp.bijectors.Identity()
        )
        shift_bijector = (
            tfp.bijectors.Shift(shift) if shift else tfp.bijectors.Identity()
        )
        return shift_bijector(scale_bijector)
    ftype = lambda x: np.array(x, dtype=default_float())
    
    # Upper/lower bound on variance of kernel.
    var_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(80.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # For SE, ACRD or APRD (base) kernels, i.e. when there is no combination of APRD and ACRD
    ls_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(100.0))(
        tfp.bijectors.Sigmoid()
    )     
    
    if arg_dict["model"] == "isotropic":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]
        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            kernel = SquaredExponential(variance=var, lengthscales=lscale_init_image)
            kernel.variance = gpflow.Parameter(kernel.variance.numpy(), transform=var_constrained())
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))
        # Initialise multi-output kernel and multi-ouput inducing variables
        kernel = SeparateIndependent(kernel_list) 
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)
        
    
    if arg_dict["model"] == "ARD":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]
        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            l_scales = tf.ones(shape=(tf.shape(Xtrain)[-1],), dtype=default_float()) * lscale_init_image
            kernel = SquaredExponential(variance=var, lengthscales=l_scales)
            kernel.variance = gpflow.Parameter(kernel.variance.numpy(), transform=var_constrained())
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))
        # Initialise multi-output kernel and multi-ouput inducing variables
        kernel = SeparateIndependent(kernel_list) 
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)
        
        
    elif arg_dict["model"] == "APRD2x2":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]
        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            base_kernel = SquaredExponential(variance=var, lengthscales=1.0)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_constrained())
            gpflow.set_trainable(base_kernel.lengthscales, False)
            aprd_kernel = APRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape= [2, 2],
                colour_channels=arg_dict["colour_channels"]
            )
            aprd_kernel.lengthscales.assign(aprd_kernel.lengthscales.numpy() * lscale_init_image)
            aprd_kernel.lengthscales = gpflow.Parameter(aprd_kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel = aprd_kernel
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))
        # Initialise multi-output kernel and multi-ouput inducing variables
        kernel = SeparateIndependent(kernel_list) 
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)
        
        
    elif arg_dict["model"] == "APRD4x4":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]
        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            base_kernel = SquaredExponential(variance=var, lengthscales=1.0)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_constrained())
            gpflow.set_trainable(base_kernel.lengthscales, False)
            aprd_kernel = APRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape= [4, 4],
                colour_channels=arg_dict["colour_channels"]
            )
            aprd_kernel.lengthscales.assign(aprd_kernel.lengthscales.numpy() * lscale_init_image)
            aprd_kernel.lengthscales = gpflow.Parameter(aprd_kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel = aprd_kernel
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))
        # Initialise multi-output kernel and multi-ouput inducing variables
        kernel = SeparateIndependent(kernel_list) 
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)
        
        
    elif arg_dict["model"] == "APRD7x7":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]
        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            base_kernel = SquaredExponential(variance=var, lengthscales=1.0)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_constrained())
            gpflow.set_trainable(base_kernel.lengthscales, False)
            aprd_kernel = APRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape= [7, 7],
                colour_channels=arg_dict["colour_channels"]
            )
            aprd_kernel.lengthscales.assign(aprd_kernel.lengthscales.numpy() * lscale_init_image)
            aprd_kernel.lengthscales = gpflow.Parameter(aprd_kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel = aprd_kernel
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))
        # Initialise multi-output kernel and multi-ouput inducing variables
        kernel = SeparateIndependent(kernel_list) 
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)



    return model, arg_dict
    


def get_optimizer(arg_dict):
    """
    Return an Adam optimizer with optional exponential learning rate decay.
    """
    lr = arg_dict["lr"]
    if arg_dict["lr_decay_steps"] > 0:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=arg_dict["lr"],
            decay_steps=arg_dict["lr_decay_steps"],
            decay_rate=arg_dict["lr_decay_factor"],
            staircase=True,
        )
    return tf.keras.optimizers.Adam(learning_rate=lr)


def has_bad_params(model):
    '''
    Model parameters become Nan when Cholesky decomposition fails.
    Also frequently occurs when variance is not clipped to a small minimum value
    when using SVGP.
    '''
    for var in model.trainable_variables:
        if not np.all(np.isfinite(var.numpy())):
            return True
    return False


def train_and_save(model, data, arg_dict):

    # Iterator over minibatches
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(model.num_data)
    train_iter = iter(train_dataset.batch(arg_dict["mb_size"]))
    
    training_loss = model.training_loss_closure(train_iter, compile=True)

    optimizer = get_optimizer(arg_dict)

    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            loss = training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Setup directory for saving
    save_dir = arg_dict["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Setup checkpointing
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=None)

    # Save arg_dict
    with open(os.path.join(save_dir, "arg_dict.json"), "w") as f:
        json.dump(arg_dict, f, indent=4, default=str)

    # Save version log
    log_versions(save_dir)

    time_log = []
    start_time = time.time()

    nan_detected = False

    for step in range(1, arg_dict["max_steps"] + 1):
        optimization_step()

        if arg_dict["max_time"] is not None and (time.time() - start_time) > arg_dict["max_time"]:
            elapsed = time.time() - start_time
            ckpt_manager.save(checkpoint_number=step)
            time_log.append({"step": step, "elapsed_seconds": elapsed})
            with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                    json.dump(time_log, f, indent=4)
            print(f"Max time {arg_dict['max_time']}s reached, stopping training.")
            print(f"Last step {step}: elapsed = {elapsed:.1f}s, checkpoint saved.")
            break

        if step % arg_dict["checkpoint_interval"] == 0:
            # Check for NaNs
            if has_bad_params(model):
                elapsed = time.time() - start_time
                time_log.append({"step": step, "elapsed_seconds": elapsed, "nan_detected": True})
                with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                    json.dump(time_log, f, indent=4)
                print(f"NaN detected in model parameters at step {step}. Stopping training.")
                nan_detected = True
                break

            elapsed = time.time() - start_time
            ckpt_manager.save(checkpoint_number=step)
            time_log.append({"step": step, "elapsed_seconds": elapsed})
            if arg_dict["verbose"]:
                print(f"Step {step}: elapsed = {elapsed:.1f}s, checkpoint saved.")

            with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                json.dump(time_log, f, indent=4)

    # Save timing log
    #with open(os.path.join(save_dir, "time_log.json"), "w") as f:
    #    json.dump(time_log, f, indent=4)

    if nan_detected:
        print(f"Training stopped early due to NaNs. Logs saved in {save_dir}")
    else:
        print(f"Training finished. Logs saved in {save_dir}")


def log_versions(save_dir):
    versions = {
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "gpflow": gpflow.__version__,
    }
    with open(os.path.join(save_dir, "version_log.json"), "w") as f:
        json.dump(versions, f, indent=4)


def main():
    arg_dict = parse_args()

    # Set default float and jitter
    gpflow.config.set_default_float(arg_dict["default_dtype"])
    gpflow.config.set_default_jitter(arg_dict["default_jitter"])

    # Set random seed
    np.random.seed(arg_dict["rseed"])
    tf.random.set_seed(arg_dict["rseed"])

    # Load data
    Xtrain, Ytrain, Xtest, Ytest, arg_dict = load_data(arg_dict)

    # Build model
    model, arg_dict = build_model(Xtrain, arg_dict)

    # Train + save
    train_and_save(model, (Xtrain, Ytrain), arg_dict)

if __name__ == "__main__":
    main()
