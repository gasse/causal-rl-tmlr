import os
import sys
import pathlib
import json
import argparse
import numpy as np
import gzip
import pickle
import torch
import tempfile


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

    args = parser.parse_args()

    print(f"")
    print(f"GENERATE OBSERVATIONAL DATA")
    print(f"experiment: {args.experiment}")
    print(f"privileged_policy: {args.expert}")
    print(f"seed: {args.seed}")
    print(f"")


    ## READ EXPERIMENT CONFIG ##

    with open(f"experiments/{args.experiment}/config.json", "r") as f:
        cfg = json.load(f)

    with open(f"experiments/{args.experiment}/hyperparams.json", "r") as f:
        params = json.load(f)

    assert args.expert in cfg['privileged_policies'].keys()

    # Pre-emptive code
    data_obs_file = pathlib.Path(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/data_obs.gz")
    if data_obs_file.exists():
        print(f"{data_obs_file} already exists, nothing to do...")
        exit(0)


    ## SET UP THE EXPERIMENT ##

    # Ugly import hack
    sys.path.insert(0, os.path.abspath(f"."))

    from environments import TabularPOMDP
    from agents import PrivilegedAgent
    from utils import construct_dataset


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(args.seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training_model = rng.randint(0, 2**10)
    seed_training_agent = rng.randint(0, 2**10)
    seed_evaluation = rng.randint(0, 2**10)

    torch.manual_seed(seed_data_obs)


    ## GENERATE THE OBSERVATIONAL DATASET ##

    # Environment (POMDP dynamics)
    env = TabularPOMDP(p_s=torch.tensor(cfg['p_s']),
        p_o_s=torch.tensor(cfg['p_o_s']),
        p_r_s=torch.tensor(cfg['p_r_s']),
        p_d_s=torch.tensor(cfg['p_d_s']),
        p_s_sa=torch.tensor(cfg['p_s_sa']))

    # Privileged policy (observational regime)
    obs_policy = PrivilegedAgent(p_a_s=torch.tensor(cfg['privileged_policies'][args.expert]))

    data_obs = construct_dataset(env=env,
        max_episode_length=cfg["max_episode_length"],
        policy=obs_policy,
        n_samples=params['nsamples_obs'],
        privileged=True)

    data_obs_file.parent.mkdir(parents=True, exist_ok=True)

    # write to temp file then move for atomicity
    with tempfile.NamedTemporaryFile(prefix=data_obs_file.name, dir=data_obs_file.parent, delete=False) as f:
        tmp_file = f.name

    print(f"Writing to temporary file {tmp_file}")
    with gzip.open(tmp_file, "wb") as f:
        pickle.dump(data_obs, f)

    print(f"Moving file to {data_obs_file}")
    try:
        os.link(tmp_file, data_obs_file)
    except OSError:
        print(f"File {data_obs_file} already there, giving up")
        pass
    os.unlink(tmp_file)
