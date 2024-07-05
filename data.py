import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def generate_linear_regression_data(n, p, x_cov_matrix, beta, mu, sigma):
    L = np.linalg.cholesky(x_cov_matrix)
    z = rng.normal(size=(p, n))
    x = L@z
    x = x.T
    eps = sigma * rng.normal(size=n)
    y = x@beta + eps + mu
    return {
        "n": n,
        "p": p,
        "x": x.tolist(),
        "y": y.tolist(),
        "beta": beta,
    }

def generate_data_for_scenario_1(n):
    p = 8
    rho = 0.5
    axis_0, axis_1 = np.indices((p, p))
    x_cov_matrix = rho**np.abs(axis_0 - axis_1)
    beta = np.array([3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    mu = 0.0
    sigma = 3.0
    return generate_linear_regression_data(n, p, x_cov_matrix, beta, mu, sigma)

def generate_data_for_scenario_2(n):
    p = 8
    rho = 0.5
    axis_0, axis_1 = np.indices((p, p))
    x_cov_matrix = rho**np.abs(axis_0 - axis_1)
    beta = 0.85 * np.ones(p)
    mu = 0.0
    sigma = 3.0
    return generate_linear_regression_data(n, p, x_cov_matrix, beta, mu, sigma)

def generate_data_for_scenario_3(n):
    p = 8
    rho = 0.5
    axis_0, axis_1 = np.indices((p, p))
    x_cov_matrix = rho**np.abs(axis_0 - axis_1)
    beta = np.zeros(p)
    beta[0] = 5.0
    mu = 0.0
    sigma = 2.0
    return generate_linear_regression_data(n, p, x_cov_matrix, beta, mu, sigma)

def generate_data_for_scenario_4(n):
    p = 40
    x_cov_matrix = 0.5 * np.ones((p, p)) + 0.5 * np.eye(p)
    beta = np.concatenate([
        np.zeros(10), 2*np.ones(10), np.zeros(10), 2*np.ones(10)
    ])
    mu = 0.0
    sigma = 15.0
    return generate_linear_regression_data(n, p, x_cov_matrix, beta, mu, sigma)

def generate_diabetes_data():
    data = pd.read_csv("diabetes.txt", sep="\t")
    x, y = data.drop("Y", axis=1), data["Y"]
    n = x.shape[0]
    p = x.shape[1]
    return {
        "n": n,
        "p": p,
        "x": x.to_numpy().tolist(),
        "y": y.to_numpy().tolist(),
        # "covariates": list(x.columns),
    }


if __name__ == "__main__":
    print(generate_diabetes_data())