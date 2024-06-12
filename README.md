# contextual_regression_issues

Contextual regression on toy datasets

In this repo, we train a contextual regression model on a toy dataset
with a known relationship between inputs and outputs. We show that
contextual regression incorrectly identifies features not actually
related to the output as important while downweighting those which
in fact have a strong relationship to the output. Notice that if you
change the NN architecture, the training hyperparameters and/or the
data generation procedure, the results of the contextual regression
procedure will change -- most typically they will still be incorrect,
but in a different way.

As illustrated here, on a simple toy problem with a known solution,
contextual regression can fail to identify important sequence positions
and instead identify randomized and/or fixed sequence positions as
more important than those with an actual relationship to the output.
Its usefulness and accuracy may therefore be NN-architecture and hyperparameter
dependent.
