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
    print(f"EVALUATE")
    print(f"experiment: {args.experiment}")
    print(f"privileged_policy: {args.expert}")
    print(f"training_scheme: {args.method}")
    print(f"nsamples_int: {args.nints}")
    print(f"seed: {args.seed}")
    print(f"gpu: {args.gpu}")
    print(f"")


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(args.seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training_model = rng.randint(0, 2**10)
    seed_training_agent = rng.randint(0, 2**10)
    seed_evaluation = rng.randint(0, 2**10)


    ## READ EXPERIMENT CONFIG ##

    with open(f"experiments/{args.experiment}/config.json", "r") as f:
        cfg = json.load(f)

    with open(f"experiments/{args.experiment}/hyperparams.json", "r") as f:
        params = json.load(f)

    assert args.expert in cfg['privileged_policies'].keys()
    assert args.method in params['training_schemes']
    assert args.nints in params['nsamples_int']


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
    from models import TabularAugmentedModel, evaluate_model
    from agents import BeliefStateActorCriticAgent, evaluate_agent
    from utils import print_log

    # Environment (POMDP dynamics)
    env = TabularPOMDP(p_s=torch.tensor(cfg['p_s']),
                   p_o_s=torch.tensor(cfg['p_o_s']),
                   p_r_s=torch.tensor(cfg['p_r_s']),
                   p_d_s=torch.tensor(cfg['p_d_s']),
                   p_s_sa=torch.tensor(cfg['p_s_sa']))
    env = env.to(device)


    ## EVALUATE THE MODEL + AGENT ##

    torch.manual_seed(seed_evaluation)

    checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{args.nints}")
    checkpointdir.mkdir(parents=True, exist_ok=True)

    # Pre-emptive code
    eval_file = checkpointdir / f"eval_model_agent.gz"
    if eval_file.exists():
        print(f"{eval_file} already exists, nothing to do...")
        exit(0)

    logfile = checkpointdir / f"log_06_eval.txt"
    print_log(f"", logfile=logfile)
    print_log(f"seed: {args.seed}", logfile=logfile)
    print_log(f"privileged_policy: {args.expert}", logfile=logfile)
    print_log(f"training_scheme: {args.method}", logfile=logfile)
    print_log(f"nsamples_obs: {params['nsamples_obs']}", logfile=logfile)
    print_log(f"nsamples_int: {args.nints}", logfile=logfile)
    print_log(f"saving results to: {checkpointdir}", logfile=logfile)

    # Test data for model evaluation
    with gzip.open(f"experiments/{args.experiment}/results/seed_{args.seed}/data_int_test.gz", "rb") as f:
        test_data = pickle.load(f)

    # Model
    model = TabularAugmentedModel(
        s_nvals=params["model_latent_space_size"],
        o_nvals=env.o_nvals,
        a_nvals=env.a_nvals,
        r_nvals=env.r_nvals,
        with_done=params["model_with_done"])
    model = model.to(device)

    # Agent
    agent = BeliefStateActorCriticAgent(
        belief_model=TabularAugmentedModel(
            s_nvals=params["model_latent_space_size"],
            o_nvals=env.o_nvals,
            a_nvals=env.a_nvals,
            r_nvals=env.r_nvals,
            with_done=params["model_with_done"]),
        hidden_size=params["agent_hidden_size"])
    agent = agent.to(device)

    model.load_state_dict(torch.load(checkpointdir / f"model.pt", map_location=device))
    agent.load_state_dict(torch.load(checkpointdir / f"agent.pt", map_location=device))

    model_nll = evaluate_model(model, test_data,
        batch_size=params["model_n_samples_per_batch"])
    agent_value = evaluate_agent(env, agent,
        reward_map=torch.tensor(cfg['r_desc']),
        max_episode_length=cfg["max_episode_length"],
        n_samples=params["agent_n_samples_evaluation"],
        batch_size=params["agent_n_samples_per_batch"])

    print_log(f"model_nll: {model_nll}", logfile=logfile)
    print_log(f"agent_value: {agent_value}", logfile=logfile)

    with gzip.open(eval_file, "wb") as f:
        pickle.dump((model_nll, agent_value), f)
