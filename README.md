# Learning to branch for the Crew Pairing Problems
*Learning to branch for the Crew Pairing Problems* is an article currently under submission at *Operations Research Forum*.
This repository contains the code used to train the branching heuristics based on the strong branching policy.

You will also find our collected dataset, trained models and results of our experiements, in the folder `data_results`.

## Models
We trained 3 models :
* Linear model
* Multilayer Perceptron (MLP)
* Transformer encoder

Considering our models have to select the branching candidate among 3 possible candidates, we would be in a situation like this:

[!Branching among 3 candidates](./images/presentation.png)

### About the linear model and the MLP
Those models evaluate each candidate separately.
It could be represented like this (for the MLP):

[!Candidate evaluation for the MLP](./images/mlp.png)

### About the Transformer encoder
This model is able to evaluate all candidates at the same time.

