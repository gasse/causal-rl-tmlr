import os
import sys
import pathlib
import json
import argparse
import numpy as np
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
    print(f"TRAIN AGENT")
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
    # from models import NNAugmentedModel
    from models import TabularAugmentedModel
    from agents import BeliefStateActorCriticAgent, train_ac_agent
    from utils import print_log

    checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{args.seed}/{args.expert}/nobs_{params['nsamples_obs']}/{args.method}/nint_{args.nints}")
    checkpointdir.mkdir(parents=True, exist_ok=True)

    # Pre-emptive code
    agent_file = checkpointdir / f"agent.pt"
    if agent_file.exists():
        print(f"{agent_file} already exists, nothing to do...")
        exit(0)

    logfile = checkpointdir / f"log_05_train_agent.txt"
    print_log(f"", logfile=logfile)
    print_log(f"seed: {args.seed}", logfile=logfile)
    print_log(f"device: {device}", logfile=logfile)
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

    torch.manual_seed(seed_training_agent + args.nints)


    ## LEARN THE DREAMER AGENT ##

    # 1. recover the augmented POMDP model
    model = TabularAugmentedModel(s_nvals=params["model_latent_space_size"],
        o_nvals=torch.tensor(cfg['p_o_s']).size(1),
        a_nvals=torch.tensor(cfg['p_s_sa']).size(1),
        r_nvals=torch.tensor(cfg['p_r_s']).size(1),
        with_done=params["model_with_done"])
    model.to(device=device)
    model.load_state_dict(torch.load(checkpointdir / f"model.pt", map_location=device))

    # 2. instanciate a new agent that uses the model's belief state
    agent = BeliefStateActorCriticAgent(
        belief_model=model,
        hidden_size=params["agent_hidden_size"])
    agent = agent.to(device)

    # 3. convert the augmented POMDP model to a dream environment
    dream_params = model.get_probs()
    dream_env = TabularPOMDP(
        p_s=dream_params["p_s"],
        p_o_s=dream_params["p_o_s"],
        p_r_s=dream_params["p_r_s"],
        p_d_s=dream_params["p_d_s"],
        p_s_sa=dream_params["p_snext_sa"])
    dream_env.to(device=device)

    # 3. train the agent via actor-critic
    train_ac_agent(dream_env, agent, reward_map=torch.tensor(cfg['r_desc']),
                    max_episode_length=cfg["max_episode_length"],
                    lr=params["agent_lr"],
                    gamma=params["agent_gamma"],
                    n_epochs_warmup=params["agent_n_epochs_warmup"],
                    n_epochs=params["agent_n_epochs_training"],
                    epoch_size=params["agent_n_batches_per_epoch"],
                    batch_size=params["agent_n_samples_per_batch"],
                    critic_weight=params["agent_critic_weight"],
                    entropy_bonus=params["agent_entropy_bonus"],
                    scale_returns=params["agent_scale_returns"],
                    log=True,
                    logfile=logfile)

    torch.save(agent.state_dict(), agent_file)
