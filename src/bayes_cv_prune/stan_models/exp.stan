data {
    int<lower=0> N; // Number of accuracies
    real a[N]; // Accuracy values
}
parameters {
    // Model parameters
    real<lower=0> mu;
    real<lower=0, upper=1> eta;

    // Uniformly distributed hyperpriors (except for beta_mu0)
    // (High "infinite" upper bound for convergence in the approximation)
    real<lower=0, upper=1e5> alpha_mu0;
    real<lower=0> beta_mu0;
    real<lower=0, upper=1e5> alpha_eta0;
    real<lower=0, upper=1e5> beta_eta0;
}
transformed parameters {
    // Beta model original parameters
    real alpha = mu * eta;
    real beta = mu * (1 - eta);
}
model {
    beta_mu0 ~ exponential(0.001);
    mu ~ gamma(alpha_mu0, beta_mu0);
    eta ~ beta(alpha_eta0, beta_eta0);
    a ~ beta(alpha, beta);
}
generated quantities {
    // Sample posterior predictive after fit
    real a_hat;
    a_hat = beta_rng(alpha, beta);
}
