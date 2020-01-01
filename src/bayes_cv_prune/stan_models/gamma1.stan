data {
    int<lower=0> N; // Number of accuracies
    real a[N]; // accuracy values
    // https://stats.stackexchange.com/questions/94303/hierarchical-bayesian-modeling-of-incidence-rates
    // https://www.researchgate.net/post/Priors_for_beta_distribution

}
parameters {
    // --- model parameters ---
    // beta distribution parameters
    real<lower=0, upper=100000> alpha;
    real<lower=0, upper=100000> beta;

    // --- model hyperparameters --- (bounded for easier convergence)
    // gamma-parameters for alpha
    real<lower=0> alpha0;
    real<lower=0> alpha1;
    // gamma-parameters for beta
    real<lower=0> beta0;
    real<lower=0> beta1;
}
model {
    alpha ~ gamma(alpha0, alpha1);
    beta ~ gamma(beta0, beta1);
    a ~ beta(alpha, beta);
}
generated quantities {
    # Sample posterior predictive after fit
    real a_hat;
    a_hat = beta_rng(alpha, beta);
}
