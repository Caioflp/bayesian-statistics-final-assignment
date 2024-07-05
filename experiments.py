import random

import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path
from cmdstanpy import CmdStanModel

from data import (
    generate_data_for_scenario_1,
    generate_data_for_scenario_2,
    generate_data_for_scenario_3,
    generate_data_for_scenario_4,
    generate_diabetes_data,
)


SEED = 42
OUTPUT_DIR = Path("outputs")


def artificial_scenarios_experiment(n_data_points=100):
    experiment_dir = OUTPUT_DIR / f"artificial_scenarios_n={n_data_points}"
    data_generation_functions = [
        generate_data_for_scenario_1,
        generate_data_for_scenario_2,
        generate_data_for_scenario_3,
        generate_data_for_scenario_4
    ]
    for i, generate_data in enumerate(data_generation_functions, start=1):
        scenario_dir = experiment_dir / f"scenario_{i}"
        stan_csv_dir = scenario_dir / "csv"
        if stan_csv_dir.exists():
            shutil.rmtree(stan_csv_dir)
        stan_csv_dir.mkdir(parents=True)

        if i == 4:
            data = generate_data(n_data_points * 5)
        else:
            data = generate_data(n_data_points)
        data["delta"], data["r"] = 1.0, 0.1 # weakly informative exponential prior
        stan_file = Path("stan/lasso.stan")
        model = CmdStanModel(stan_file=stan_file)
        fit = model.sample(
            data,
            chains=5,
            seed=SEED,
            iter_sampling=10_000,
        )
        fit.save_csvfiles(stan_csv_dir)

        # Save summaries
        summary = fit.summary(percentiles=(2.5, 50, 97.5), sig_figs=4)
        with open(scenario_dir / "summary.tex", "w") as file:
            summary_to_save = summary.loc["lp__":"lambda", "N_Eff":"R_hat"].drop("N_Eff/s", axis=1)
            summary_to_save.columns = [column.replace("%", "\%") for column in summary_to_save.columns]
            summary_to_save.to_latex(file)

        # Create traceplots
        if i != 4: # scenario 4 has 40 covariates
            fig, axs = plt.subplots(nrows=(data["p"]+2)//2, ncols=2, figsize=(7, 10), sharex=True)
            axs = axs.flatten()
            parameters_samples = {}
            parameters_samples["sigma"] = fit.stan_variable("sigma").reshape(5, 10_000)
            parameters_samples["lambda"] = fit.stan_variable("lambda").reshape(5, 10_000)
            for j in range(0, data["p"]):
                parameters_samples[f"beta[{j+1}]"] = fit.stan_variable("beta")[:, j].reshape(5, 10_000)
            for ax, (parameter, samples) in zip(axs, parameters_samples.items()):
                ax.plot(samples[random.randint(0, 4)], linewidth=0.5)
                ax.set_title(parameter)
                ax.set_xlabel("iteration")
            fig.suptitle("Trace plots for parameters")
            fig.tight_layout()
            fig.savefig(scenario_dir / "traceplots.pdf")
        
        # Plot posterior predictive check
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), sharex=True, sharey=True)
        iterations = random.sample(list(range(0, 40_000)), k=5)
        y_versions = [data["y"]]
        for j in iterations:
            y_versions.append(fit.stan_variable("y_pred")[j])
        for ax, y in zip(axs.flatten(), y_versions):
            ax.hist(y)
            ax.set_xlabel("y")
        axs.flatten()[0].set_title("Data")
        for j, iteration in enumerate(iterations, start=1):
            axs.flatten()[j].set_title(f"Iteration {iteration%10_000} from chain {iteration//10_000}")
        fig.suptitle("Posterior predictive samples of y")
        fig.savefig(scenario_dir / "posterior_predictive_check.pdf")

        # Plot 95% credibility intervals
        betas = fit.draws_pd().loc[:, "beta[1]":f"beta[{data['p']}]"]
        fig, ax = plt.subplots(figsize=(5, data["p"]//3))
        sns.boxplot(
            data=betas,
            orient="h",
            width=0.5,
            whis=(2.5, 97.5),
            fliersize=0,
            ax=ax,
            legend="auto",
            color="blue",
            gap=0.5,
            fill=False,
            zorder=1
        )
        true_beta = data["beta"]
        ax.scatter(
            x=true_beta,
            y=list(range(len(true_beta))),
            c="red",
            marker="x",
            zorder=2,
        )
        ax.axvline(x=0, linestyle="--", alpha=0.5, color="red")
        ax.set_title("95% credibility intervals")
        fig.savefig(scenario_dir / "credibility_intervals.pdf")


if __name__ == "__main__":
    artificial_scenarios_experiment(n_data_points=20)
    artificial_scenarios_experiment(n_data_points=100)