import os
import sys
import pathlib
import json
import argparse
import numpy as np
import torch
import copy

import matplotlib.pyplot as plt


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
        '--nseeds',
        type=int,
        required=False,
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        help='GPU id (-1 for CPU).',
        default=-1,
    )

    args = parser.parse_args()

    print(f"")
    print(f"PLOT DENSITIES")
    print(f"experiment: {args.experiment}")
    print(f"privileged_policy: {args.expert}")
    print(f"nseeds: {args.nseeds}")
    print(f"gpu: {args.gpu}")
    print(f"")


    ## READ EXPERIMENT CONFIG ##

    with open(f"experiments/{args.experiment}/config.json", "r") as f:
        cfg = json.load(f)

    with open(f"experiments/{args.experiment}/hyperparams.json", "r") as f:
        params = json.load(f)

    assert args.expert in cfg['privileged_policies'].keys()
    assert args.seed is None or args.nseeds is None, "--seed and --nseeds should not be specified simultaneously"
    assert not (args.seed is None and args.nseeds is None), "either --seed or --nseeds should be specified"
    assert 'grid_size' in cfg and 's_coords' in cfg, "This toy example is missing a grid configuration"


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
    from agents import BeliefStateActorCriticAgent

    # Environment (POMDP dynamics)
    env = TabularPOMDP(p_s=torch.tensor(cfg['p_s']),
                   p_o_s=torch.tensor(cfg['p_o_s']),
                   p_r_s=torch.tensor(cfg['p_r_s']),
                   p_d_s=torch.tensor(cfg['p_d_s']),
                   p_s_sa=torch.tensor(cfg['p_s_sa']))
    env = env.to(device)

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

    max_episode_length = cfg["max_episode_length"]
    n_samples_per_seed = 10000

    coords_x, coords_y = cfg["s_coords"]
    coords_x = np.asarray(coords_x)
    coords_y = np.asarray(coords_y)

    grid_visits = {}
    for nints in params['nsamples_int']:

        if args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = range(args.nseeds)


        ## COLLECT TRAJECTORIES ##

        state_visits = {method: np.zeros(env.s_nvals) for method in params['training_schemes']}
        for seed in seeds:

            # set up the seeds
            rng = np.random.RandomState(seed)
            seed_data_obs = rng.randint(0, 2**10)
            seed_data_int = rng.randint(0, 2**10)
            seed_training_model = rng.randint(0, 2**10)
            seed_training_agent = rng.randint(0, 2**10)
            seed_evaluation = rng.randint(0, 2**10)

            for method in params['training_schemes']:

                print(f"generating trajectories: nints {nints} seed {seed} method {method}")

                torch.manual_seed(seed_evaluation)

                # load agent parameters
                checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{seed}/{args.expert}/nobs_{params['nsamples_obs']}/{method}/nint_{nints}")
                agent.load_state_dict(torch.load(checkpointdir / f"agent.pt", map_location=device))

                # collect trajectories
                agent.reset()
                obs, reward, done, info = env.reset(n_samples_per_seed)
                t = 0

                # count grid tile visits
                was_done = torch.full_like(done.to(dtype=torch.bool), False)
                state_visits[method] += np.bincount(info["state"][was_done.logical_not()].cpu().numpy(), minlength=env.s_nvals)

                while True:

                    # stop if maximum episode length reached
                    if t == max_episode_length:
                        break

                    # early stop if all episodes have ended
                    if all(done):
                        break

                    action = agent.action(obs, reward, done)
                    obs, reward, done, info = env.step(action)
                    t += 1

                    # count grid tile visits
                    was_done = torch.logical_or(done.to(dtype=torch.bool), was_done)  # enforce persistent done flag after raised (should be true)
                    state_visits[method] += np.bincount(info["state"][was_done.logical_not()].cpu().numpy(), minlength=env.s_nvals)


        ## PLOT GRID VISIT DENSITY ##

        plotsdir = pathlib.Path(f"experiments/{args.experiment}/plots")
        plotsdir.mkdir(parents=True, exist_ok=True)

        grid_visits[nints] = {}
        for method in params['training_schemes']:
            grid_visits[nints][method] = np.zeros(cfg['grid_size'])
            for state, visits in enumerate(state_visits[method]):
                grid_visits[nints][method][coords_x[state], coords_y[state]] += visits

    grid_visits_max = np.max([[visits for visits in x.values()] for x in grid_visits.values()])

    for nints in params['nsamples_int']:
        for method in params['training_schemes']:
            if args.seed is not None:
                plotfile = plotsdir / f"{args.expert}_densities_nobs_{params['nsamples_obs']}_seed_{args.seed}_nint_{nints}_method_{method}.pdf"
            else:
                plotfile = plotsdir / f"{args.expert}_densities_nobs_{params['nsamples_obs']}_nint_{nints}_method_{method}.pdf"

            print(f"writing figure {plotfile}")

            fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)
            ax.imshow(grid_visits[nints][method].transpose(), vmin=0, vmax=grid_visits_max, origin='lower', cmap='Blues', interpolation='nearest')
            ax.axis('off')
            ax.set_aspect('equal')
            fig.savefig(plotfile, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
