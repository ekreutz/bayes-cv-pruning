data {
    int<lower=0> N; // Number of accuracies
    real a[N]; // accuracy values
    vector<lower=0>[2] mu0_upper;
    vector<lower=0>[2] eta0_upper;
}
parameters {
    // --- model parameters ---
    real<lower=0> mu;
    real<lower=0, upper=1> eta;

    // hyperparameters for prior
    real<lower=0, upper=mu0_upper[1]> alpha_mu0;
    real<lower=0, upper=mu0_upper[2]> beta_mu0;
    real<lower=0, upper=eta0_upper[1]> alpha_eta0;
    real<lower=0, upper=eta0_upper[2]> beta_eta0;
}
transformed parameters {
    // Beta parameters
    real alpha = mu * eta;
    real beta = mu * (1 - eta);
}
model {
    mu ~ gamma(alpha_mu0, beta_mu0);
    eta ~ beta(alpha_eta0, beta_eta0);
    a ~ beta(alpha, beta);
}
generated quantities {
    // Sample posterior predictive after fit
    real a_hat;
    a_hat = beta_rng(alpha, beta);
}
