data {
    int<lower=0> N; // Number of accuracies
    real a[N]; // Accuracy values
}
parameters {
    // Model parameters
    real<lower=0> mu;
    real<lower=0, upper=1> eta;
}
transformed parameters {
    // Beta model original parameters
    real alpha = mu * eta;
    real beta = mu * (1 - eta);
}
model {
    mu ~ exponential(0.01);
    eta ~ beta(1, 1);
    a ~ beta(alpha, beta);
}
generated quantities {
    // Sample posterior predictive after fit
    real a_hat;
    a_hat = beta_rng(alpha, beta);
}
