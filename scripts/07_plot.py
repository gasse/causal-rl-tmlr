import os
import sys
import pathlib
import json
import argparse
import gzip
import pickle
import numpy as np
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
        '--nints_min',
        type=int,
    )
    parser.add_argument(
        '--nints_max',
        type=int,
    )

    args = parser.parse_args()

    print(f"")
    print(f"PLOT")
    print(f"")
    print(f"experiment: {args.experiment}")
    print(f"privileged_policy: {args.expert }")
    print(f"nseeds: {args.nseeds}")
    print(f"nints_min: {args.nints_min}")
    print(f"nints_max: {args.nints_max}")


    ## READ EXPERIMENT CONFIG ##

    with open(f"experiments/{args.experiment}/config.json", "r") as f:
        cfg = json.load(f)

    with open(f"experiments/{args.experiment}/hyperparams.json", "r") as f:
        params = json.load(f)

    assert args.expert in cfg['privileged_policies'].keys()
    assert args.seed is None or args.nseeds is None, "--seed and --nseeds should not be specified simultaneously"
    assert not (args.seed is None and args.nseeds is None), "either --seed or --nseeds should be specified"

    nsamples_int_subsets = params['nsamples_int']

    nsamples_int_subsets = np.asarray(nsamples_int_subsets)
    if args.nints_min is not None:
        nsamples_int_subsets = nsamples_int_subsets[nsamples_int_subsets >= args.nints_min]
    if args.nints_max is not None:
        nsamples_int_subsets = nsamples_int_subsets[nsamples_int_subsets <= args.nints_max]


    ## COLLECT THE RESULTS ##

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = range(args.nseeds)

    likelihoods = {}
    returns = {}
    for training_scheme in params["training_schemes"]:
        likelihoods[training_scheme] = []
        returns[training_scheme] = []

        for seed in seeds:
            seed_likelihoods = []
            seed_rewards = []

            for nsamples_int in nsamples_int_subsets:
                checkpointdir = pathlib.Path(f"experiments/{args.experiment}/results/seed_{seed}/{args.expert}/nobs_{params['nsamples_obs']}/{training_scheme}/nint_{nsamples_int}")

                print(f"Reading results from {checkpointdir}")

                with gzip.open(checkpointdir / f"eval_model_agent.gz", "rb") as f:
                    model_nll, agent_return = pickle.load(f)
                    seed_likelihoods.append(np.exp(-model_nll))
                    seed_rewards.append(agent_return)

            likelihoods[training_scheme].append(seed_likelihoods)
            returns[training_scheme].append(seed_rewards)

        likelihoods[training_scheme] = np.asarray(likelihoods[training_scheme])
        returns[training_scheme] = np.asarray(returns[training_scheme])


    ## CREATE AND SAVE THE PLOTS ##

    # Ugly hack
    sys.path.insert(0, os.path.abspath(f"."))

    from utils import compute_central_tendency_and_error, plot_mean_lowhigh

    labels = {
        "int": "no obs",
        "obs+int": "naive",
        "augmented": "augmented",}
    colors = {
        "int": "tab:blue",
        "obs+int": "tab:orange",
        "augmented": "tab:green",}
    markers = {
        "int": "v",
        "obs+int": "s",
        "augmented": None,}

    test = 'Wilcoxon'
    deviation = 'std' #'sem'
    confidence_level = 0.05

    plotsdir = pathlib.Path(f"experiments/{args.experiment}/plots")
    plotsdir.mkdir(parents=True, exist_ok=True)

    rmin = np.min([r for r in returns.values()])
    rmax = np.max([r for r in returns.values()])

    likmin = np.min([l for l in likelihoods.values()])
    likmax = np.max([l for l in likelihoods.values()])

    ### Model likelihood ###

    fig, axes = plt.subplots(1, 1, figsize=(3, 2.25), dpi=300)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    ax = axes

    for training_scheme in params["training_schemes"]:
        if args.nseeds > 1:
            mean, low, high = compute_central_tendency_and_error('mean', deviation, likelihoods[training_scheme])
            plot_mean_lowhigh(ax, nsamples_int_subsets, mean, low, high, label=labels[training_scheme], color=colors[training_scheme])
        else:
            ax.plot(nsamples_int_subsets, likelihoods[training_scheme][0], label=labels[training_scheme], color=colors[training_scheme])

    ax.set_title(f"Likelihood")
    ax.set_xlabel(r'$|\mathcal{D}_{std}|$')
    ax.set_ylim(bottom=0)

    fig.savefig(plotsdir / f"{args.expert}_lik_nobs_{params['nsamples_obs']}.pdf", bbox_inches='tight', pad_inches=0)

    ax.legend()
    fig.savefig(plotsdir / f"{args.expert}_lik_nobs_{params['nsamples_obs']}_legend.pdf", bbox_inches='tight', pad_inches=0)

    plt.close(fig)


    ### Reward ###

    fig, axes = plt.subplots(1, 1, figsize=(3, 2.25), dpi=300)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    ax = axes

    for training_scheme in params["training_schemes"]:
        if args.nseeds > 1:
            mean, low, high = compute_central_tendency_and_error('mean', deviation, returns[training_scheme])
            plot_mean_lowhigh(ax, nsamples_int_subsets, mean, low, high, label=labels[training_scheme], color=colors[training_scheme])
        else:
            ax.plot(nsamples_int_subsets, returns[training_scheme][0], label=labels[training_scheme], color=colors[training_scheme])

    ax.set_title(f"Cumulated reward")
    ax.set_xlabel(r'$|\mathcal{D}_{std}|$')

    fig.savefig(plotsdir / f"{args.expert}_reward_nobs_{params['nsamples_obs']}.pdf", bbox_inches='tight', pad_inches=0)

    ax.legend()
    fig.savefig(plotsdir / f"{args.expert}_reward_nobs_{params['nsamples_obs']}_legend.pdf", bbox_inches='tight', pad_inches=0)

    plt.close(fig)
