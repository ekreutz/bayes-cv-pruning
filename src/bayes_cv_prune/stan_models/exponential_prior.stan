data {
    int<lower=0> N; // Number of accuracies
    real a[N]; // accuracy values
}
parameters {
    // --- model parameters ---
    // beta distribution parameters
    real<lower=0> alpha;
    real<lower=0> beta;

    // --- model hyperparameters ---
    real<lower=0> gamma0;
    real<lower=0> gamma1;
}
model {
    alpha ~ exponential(gamma0);
    beta ~ exponential(gamma1);
    a ~ beta(alpha, beta);
}
generated quantities {
    # Sample posterior predictive after fit
    real a_hat;
    a_hat = beta_rng(alpha, beta);
}
