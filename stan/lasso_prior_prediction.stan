data{
    real<lower=0> r; // lambda hyperprior parameter
    real<lower=0> delta; // lambda hyperprior parameter
    int<lower=0> n;
    int<lower=0> p;
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
}
parameters {
    real mu; // intercept
    vector[p] beta; // predictors
    real<lower=0> sigma; // error scale
    real<lower=0> lambda; // LASSO parameter
}
model {
    target += - log(sigma); // improper 1/sigma prior
    lambda ~ gamma(delta, r);
    target += n * ( log(lambda) - log(sigma) ) - ( lambda * norm1(beta) / sigma ); // beta laplace prior
    y ~ normal(mu + standardized_x * beta, sigma);
}