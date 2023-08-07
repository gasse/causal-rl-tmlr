import os
import sys
import pathlib
import json
import argparse
import numpy as np
import gzip
import pickle
import torch


if __name__ == '__main__':

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--expert',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help = 'Random generator seed.',
        default=0,
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--nints',
        type=int,
        help = 'Number of interventional samples.',
        required=True,
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        help='GPU id (-1 for CPU).',
        default=-1,
    )

    args = parser.parse_args()

    print(f"")
    print(f"TRAIN MODEL")
    print(f"experiment: {args.experiment}")
    print(f"privileged_policy: {args.expert}")
    print(f"training_scheme: {args.method}")
    print(f"nsamples_int: {args.nints}")
    print(f"seed: {args.seed}")
    print(f"gpu: {args.gpu}")
    print(f"")


    ## READ EXPERIMENT CONFIG ##

    with open(f"experiments/{args.experiment}/config.json", "r") as f:
        cfg = json.load(f)

    with open(f"experiments/{args.experiment}/hyperparams.json", "r") as f:
        params = json.load(f)

    assert args.expert in cfg['privileged_policies'].keys()
    assert args.method in params['training_schemes']
    assert args.nints in params['nsamples_int']

    assert all(np.asarray(params['nsamples_int']) == np.unique(params['nsamples_int']))


    ## SET UP THE EXPERIMENT ##

    if args.gpu == -1:
        device = "cpu"
    elif torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.backends.mps.is_available():
        device = f"mps:{args.gpu}"
    else:
        raise Exception("No CUDA or MPS GPU available")

    # Ugly import hack
    sys.path.insert(0, os.path.abspath(f"."))

    from models import TabularAugmentedModel, fit_model
    from utils import print_log


    checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{args.nints}")
    checkpointdir.mkdir(parents=True, exist_ok=True)

    # Pre-emptive code
    model_file = checkpointdir / f"model.pt"
    if model_file.exists():
        print(f"{model_file} already exists, nothing to do...")
        exit(0)

    logfile = checkpointdir / f"log_04_train_model.txt"
    print_log(f"", logfile=logfile)
    print_log(f"seed: {args.seed}", logfile=logfile)
    print_log(f"privileged_policy: {args.expert}", logfile=logfile)
    print_log(f"training_scheme: {args.method}", logfile=logfile)
    print_log(f"nsamples_obs: {params['nsamples_obs']}", logfile=logfile)
    print_log(f"nsamples_int: {args.nints}", logfile=logfile)
    print_log(f"saving results to: {checkpointdir}", logfile=logfile)


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(args.seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training_model = rng.randint(0, 2**10)
    seed_training_agent = rng.randint(0, 2**10)
    seed_evaluation = rng.randint(0, 2**10)

    torch.manual_seed(seed_training_model + args.nints)


    ## COLLECT DATA ##

    # Recover observational dataset
    with gzip.open(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/data_obs.gz", "rb") as f:
        data_obs = pickle.load(f)

    # Recover interventional dataset
    data_int = []
    for nsamples_int in params['nsamples_int']:

        with gzip.open(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{nsamples_int}/new_data_int.gz", "rb") as f:
            data_int += pickle.load(f)

        if nsamples_int == args.nints:
            break

        assert len(data_int) == nsamples_int


    ## LEARN THE MODEL ##

    # prepare training data
    if args.method == 'int':
        train_data = data_int
    elif args.method == 'obs+int':
        train_data = [(torch.ones_like(regime), episode) for (regime, episode) in data_obs] + data_int
    elif args.method == 'augmented':
        train_data = data_obs + data_int
    else:
        raise NotImplemented

    # instanciate new model
    model = TabularAugmentedModel(s_nvals=params["model_latent_space_size"],
        o_nvals=torch.tensor(cfg['p_o_s']).size(1),
        a_nvals=torch.tensor(cfg['p_s_sa']).size(1),
        r_nvals=torch.tensor(cfg['p_r_s']).size(1),
        with_done=params["model_with_done"])
    model = model.to(device=device)

    # fit new model
    if train_data:
        fit_model(model,
                train_data=train_data,
                valid_data=train_data,  # we want to overfit
                loss_type=params["model_loss_type"],
                n_epochs=params["model_n_epochs"],
                epoch_size=params["model_n_batches_per_epoch"],
                batch_size=params["model_n_samples_per_batch"],
                lr=params["model_lr"],
                patience=params["model_patience"],
                log=True,
                logfile=logfile)

    torch.save(model.state_dict(), model_file)
