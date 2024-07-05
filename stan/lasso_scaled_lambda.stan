data{
    real<lower=0> r; // lambda hyperprior parameter
    real<lower=0> delta; // lambda hyperprior parameter
    int<lower=1> n;
    int<lower=1> p;
    matrix[n, p] x;
    vector[n] y;
}
transformed data{
    matrix[n, p] standardized_x;
    row_vector[p] mean_x; // covariates means
    row_vector[p] std_x; // covariates stds
    for (j in 1:p){
        mean_x[j] = mean(col(x, j));
        std_x[j] = sd(col(x, j));
    }
    for (i in 1:n){
        standardized_x[i] = ( row(x, i) - mean_x) ./ std_x;
    }

    real norm_standardized_x; // L_infty norm of X matrix
    norm_standardized_x = 0;
    real row_norm;
    for (i in 1:n){
        row_norm = norm1(row(standardized_x, i));
        if (norm_standardized_x < row_norm) norm_standardized_x = row_norm;
    }
}
parameters {
    real mu; // intercept
    vector[p] beta; // predictors
    real<lower=0> sigma; // error scale
    real<lower=0> lambda; // LASSO parameter
}
transformed parameters {
    real<lower=0> scaled_lambda;
    scaled_lambda = ( 2 * sqrt(log(p)) * norm_standardized_x ) * lambda;
    print("X norm: ", norm_standardized_x);
    print("Lambda ", lambda);
    print("p ", p);
    print("Scaled lambda: ", scaled_lambda);
}
model {
    target += - log(sigma); // improper 1/sigma prior
    lambda ~ gamma(delta, r);
    target += n * ( log(scaled_lambda) - log(sigma) ) - ( scaled_lambda * norm1(beta) / sigma ); // beta laplace prior
    y ~ normal(mu + standardized_x * beta, sigma);
}