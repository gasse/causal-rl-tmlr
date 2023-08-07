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
    print(f"GENERATE NEW INTERVENTIONAL DATA")
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

    from environments import TabularPOMDP
    from models import TabularAugmentedModel
    from agents import BeliefStateActorCriticAgent, EpsilonRandomAgent, UniformAgent
    from utils import construct_dataset


    checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{args.nints}")
    checkpointdir.mkdir(parents=True, exist_ok=True)

    # Pre-emptive code
    new_data_int_file = checkpointdir / f"new_data_int.gz"
    if new_data_int_file.exists():
        print(f"{new_data_int_file} already exists, nothing to do...")
        exit(0)


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(args.seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training_model = rng.randint(0, 2**10)
    seed_training_agent = rng.randint(0, 2**10)
    seed_evaluation = rng.randint(0, 2**10)

    torch.manual_seed(seed_data_int + args.nints)


    ## GENERATE NEW INTERVENTIONAL DATA ##

    # Environment (POMDP dynamics)
    env = TabularPOMDP(p_s=torch.tensor(cfg['p_s']),
                   p_o_s=torch.tensor(cfg['p_o_s']),
                   p_r_s=torch.tensor(cfg['p_r_s']),
                   p_d_s=torch.tensor(cfg['p_d_s']),
                   p_s_sa=torch.tensor(cfg['p_s_sa']))
    env.to(device=device)

    # Recover last agent (if any)
    nints_pos = np.searchsorted(params['nsamples_int'], args.nints)
    if nints_pos == 0:
        nints_last = 0
        last_agent = UniformAgent(a_nvals=env.a_nvals)
        last_agent.to(device=device)
    else:
        nints_last = params['nsamples_int'][nints_pos - 1]
        last_agent = BeliefStateActorCriticAgent(
            belief_model=TabularAugmentedModel(s_nvals=params["model_latent_space_size"],
                o_nvals=env.o_nvals,
                a_nvals=env.a_nvals,
                r_nvals=env.r_nvals,
                with_done=params["model_with_done"]),
            hidden_size=params["agent_hidden_size"])
        last_agent = last_agent.to(device=device)
        last_agent.load_state_dict(torch.load(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{nints_last}/agent.pt", map_location=device))

    # Collect new interventional data (if needed)
    new_data_int = []
    if args.nints > nints_last:

        # force random exploration for the first X samples
        if nints_last < params["nsamples_int_full_exploration"]:
            exploration_policy = UniformAgent(a_nvals=env.a_nvals)
            exploration_policy = exploration_policy.to(device=device)
            new_data_int += construct_dataset(env=env,
                max_episode_length=cfg["max_episode_length"],
                policy=exploration_policy,
                n_samples=min(args.nints, params["nsamples_int_full_exploration"]) - nints_last,
                privileged=False)

        # then, explore with last agent (+ noise)
        if params["nsamples_int_full_exploration"] < args.nints:
            exploration_policy = EpsilonRandomAgent(last_agent, env.a_nvals, epsilon=params["exploration_noise"])
            exploration_policy = exploration_policy.to(device=device)
            new_data_int += construct_dataset(env=env,
                max_episode_length=cfg["max_episode_length"],
                policy=exploration_policy,
                n_samples=args.nints - max(nints_last, params["nsamples_int_full_exploration"]),
                privileged=False)

    assert len(new_data_int) == args.nints - nints_last

    # store data tensors on CPU memory
    new_data_int = [[regime.to(device='cpu'), [x.to(device='cpu') for x in episode]] for regime, episode in new_data_int]

    # save newly collected data
    with gzip.open(new_data_int_file, "wb") as f:
        pickle.dump(new_data_int, f)
