# `bayes-cv-pruning`

Statistically valid pruning of cross validation folds using Bayesian methods.

STAN model

- [src/bayes_cv_prune/stan_models/exp_new.stan](src/bayes_cv_prune/stan_models/exp_new.stan)

Python interface for STAN model

- [src/bayes_cv_prune/StanModel.py](src/bayes_cv_prune/StanModel.py)

Algorithm from the paper - reference implementation for toy ML model

- [src/train.py](src/train.py)

# Abstract

I present a Bayesian method to model mean validation accuracy values computed in the process of performing hyperparameter optimization when training a machine learning model. The Bayesian model is based on the Beta distribution. Further, I formulate an algorithm by which said statistical model can be exploited to perform cross validation faster, by pruning CV iterations for hyperparameters that are likely to yield poor performance.

# Conclusions and future work

In this paper I showed how Bayesian methods can successfully be used to prune CV iterations that will not yield good generalization in terms of classification accuracy. When training a machine learning model, this allows for selecting a wider hyperparameter search space, faster hyperparameter optimization overall and better utilization of available computing resources. Unlike other similar methods (***reference!!), this new method introduces a statistical model for making pruning decisions to avoid making poor decisions because of high uncertainty.

Additionally, I introduced an algorithm to take advantage of the fact that the presented Bayesian model can be used to model the objective function even after very few samples. Using this algorithm is not necessary for using the Bayesian model in pruning CV iterations, but it could make the hyperparameter optimization more efficient.

# Full report

Attached as `bayes_cv_pruning.pdf`
