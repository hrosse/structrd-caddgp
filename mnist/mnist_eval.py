import os
import json
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import default_float
from gpflow.utilities import to_default_float

from mnist_train import load_data, build_model, get_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models from checkpoints")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory where checkpoints and logs are stored")
    parser.add_argument("--result_dir", type=str, default="./results/mnist",
                        help="Directory to save results of evaluation")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Size of batches for evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return vars(parser.parse_args())



def evaluate_checkpoints(model, data, save_dir, arg_dict, verbose):
    # Setup directory for saving
    os.makedirs(arg_dict["result_dir"], exist_ok=True)
    
    Xtrain, Ytrain, Xtest, Ytest = data

    optimizer = get_optimizer(arg_dict)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # for computation of (negative) ELBO in batches:
    # First compute variational expectations per batch
    # Return sum of variational expectations of batch
    # Later: Collect values in list in loop body, sum values and subtract KL divergence
    @tf.function
    def variational_expectations(data_batch):
        X, Y = data_batch
        f_mean, f_var = model.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = model.likelihood.variational_expectations(X, f_mean, f_var, Y)
        return tf.reduce_sum(var_exp)

    # Computation of approximate posteriors of latent functions per test batch
    # Used with likelihood to compute log predictive density at test locations and determine correctness of classification
    @tf.function
    def predictf(data_batch):
        X, Y = data_batch
        f_mean, f_var = model.predict_f(X, full_cov=False, full_output_cov=False)
        return f_mean, f_var
    
    
    # Load logs
    with open(os.path.join(save_dir, "time_log.json"), "r") as f:
        time_log = json.load(f)

    step_list = list()
    elapsed_list = list()
    correct_rate_list = list()
    error_list = list()
    nlpp_list = list()
    nlpp_miss_list = list()
    nlpp_correct_list = list()
    nelbo_list = list()
    
    for entry in time_log:
        step = entry["step"]
        elapsed = entry["elapsed_seconds"]

        if "nan_detected" in entry and entry["nan_detected"]:
            print(f"Training stopped at step {step} due to NaN values. No further evaluation performed.")
            break  # exit evaluation early

        ckpt_path = os.path.join(save_dir, f"ckpt-{step}")
        if not tf.train.latest_checkpoint(ckpt_path) and not os.path.exists(ckpt_path + ".index"):
            print(f"Step {step}: checkpoint not found, skipping.")
            continue

        # Restore checkpoint
        ckpt.restore(ckpt_path).expect_partial()
        if verbose:
            print(f"Restored checkpoint at step {step} (elapsed {elapsed:.1f}s).")

        step_list.append(step)
        elapsed_list.append(elapsed)

        train_iter = iter(tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain)).batch(arg_dict["batch_size"]))
        test_iter = iter(tf.data.Dataset.from_tensor_slices((Xtest, Ytest)).batch(arg_dict["batch_size"]))

        # Compute exact elbo in batches
        varexp_list = list()
        for batch_train in train_iter:
            varexp = variational_expectations(batch_train)
            varexp_list.append(variational_expectations(batch_train))
        nelbo = -(tf.reduce_sum(varexp_list) - model.prior_kl())
        nelbo_list.append(nelbo.numpy())

        # Predict latent functions in batches
        pred_list = list()
        logdensity_list = list()
        for batch_test in test_iter:
            f_mean, f_var = predictf(batch_test)
            prediction = model.likelihood.predict_mean_and_var(batch_test[0], f_mean, f_var)[0]
            logdensity = model.likelihood.predict_log_density(batch_test[0], f_mean, f_var, batch_test[1])
            pred_list.append(prediction)
            logdensity_list.append(logdensity)
            
        # Concatenate intermediary lists
        prediction = tf.concat(pred_list, axis=0)
        logdensity = tf.concat(logdensity_list, axis=0)
        
        nlpp = -tf.reduce_sum(logdensity)
        nlpp_list.append(nlpp.numpy())
        
        # Define max prob as decision criterium
        ctrue = tf.argmax(Ytest, axis=1).numpy()
        cpred = tf.argmax(prediction, axis=1).numpy()
        c_correct = (cpred == ctrue)
        rate_correct = np.sum(c_correct)/c_correct.shape[0]
        correct_rate_list.append(rate_correct)
        error_list.append(1 - rate_correct)
        mask_miss = (c_correct == False)
        nlpp_miss = -np.sum(logdensity.numpy()[mask_miss])
        nlpp_miss_list.append(nlpp_miss)
        nlpp_correct = nlpp - nlpp_miss
        nlpp_correct_list.append(nlpp_correct.numpy())

    # Final predictions: full, correct, missclassified
    # Reuse latest tensor prediction and array mask_miss
    correct_flag = np.array([1 if c else 0 for c in c_correct])
    final_pred = prediction.numpy()
    final_pred_correct = prediction.numpy()[mask_miss == False]
    final_pred_miss = prediction.numpy()[mask_miss == True]
    
    scores = dict(
        step = step_list,
        elapsed = elapsed_list,
        nelbo = np.array(nelbo_list),
        correct_rate = np.array(correct_rate_list),
        error = np.array(error_list),
        nlpp = np.array(nlpp_list),
        nlpp_miss = np.array(nlpp_miss_list),
        nlpp_correct = np.array(nlpp_correct_list)
    )

    final_predictions = dict(
        correct_flag = correct_flag,
        final_pred = final_pred,
        final_pred_correct = final_pred_correct,
        final_pred_miss = final_pred_miss
    )

    parameter_dict = gpflow.utilities.parameter_dict(model)
    model_dict = dict()
    for key, value in parameter_dict.items():
        model_dict[key] = value.numpy()

    # --- Save results in file(s) ---
    np.savez_compressed(arg_dict["result_dir"] + "/scores.npz", **scores)
    np.savez_compressed(arg_dict["result_dir"] + "/final_predictions.npz", **final_predictions)
    np.savez_compressed(arg_dict["result_dir"] + "/model.npz", **model_dict)

    

def main():
    args = parse_args()
    save_dir = args["save_dir"]
    verbose = args["verbose"]

    # Load training args
    with open(os.path.join(save_dir, "arg_dict.json"), "r") as f:
        arg_dict = json.load(f)

    # Separate directory for each selected model
    arg_dict["result_dir"] = args["result_dir"] + "/" + arg_dict["model"]
    
    arg_dict["batch_size"] = args["batch_size"]

    # Set default float and jitter
    if arg_dict["default_dtype"] == "<class 'numpy.float32'>":
        gpflow.config.set_default_float(np.float32)
    elif arg_dict["default_dtype"] == "<class 'numpy.float64'>":
        gpflow.config.set_default_float(np.float64)
    elif arg_dict["default_dtype"] == "<class 'numpy.float16'>":
        gpflow.config.set_default_float(np.float16)
    else:
        print("no valid float detected")
    gpflow.config.set_default_jitter(arg_dict["default_jitter"])

    # Set random seed
    np.random.seed(arg_dict["rseed"])
    tf.random.set_seed(arg_dict["rseed"])

    # Load data
    Xtrain, Ytrain, Xtest, Ytest, arg_dict = load_data(arg_dict)

    # Build model
    model, arg_dict = build_model(Xtrain, arg_dict)

    evaluate_checkpoints(model, (Xtrain, Ytrain, Xtest, Ytest), save_dir, arg_dict, verbose=verbose)
    

if __name__ == "__main__":
    main()
