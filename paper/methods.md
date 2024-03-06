# Methods

In the following, we assume the following causal graph.

```mermaid
```

Where $W$ represents the set of 6 first principal components, $C$ corresponds to both Age and Genetic-Sex, $V_i$ is any genetic variant and $Y$ any trait.
In particular, we assume that the principal components are a sufficient adjustment set to identify the causal effects we are interested in. This is not so much an issue from a statistical standpoint since we will be in control of the data generating process. The most important point being that there is indeed some relationship between genetic variants, principal components and traits (see Appendix).

## Samplers

In order to generate new data that is as close as possible to true real-world data, we use the original dataset to create our samplers. We propose two strategies, one resulting in the null hypothesis of no effect and one resulting in non-null effect sizes.

### Permutation Null Sampler

In this scenario, we aim to create a simulated dataset that results in the null hypothesis of no effect. However, we would like to preserve most aspects of the original data. This is done by independently sampling from the marginal distribution of each variable in the graph. Pictorially this is equivalent to removing all the edges in the graph but still preserving the marginal distributions.

```mermaid
```

The sampling Mechanism can be summarized as follows.

Permutation Null Sampler:

1. First $W,C$ are jointly sampled with replacement, hence preserving the structure between these variables.
2. Each genetic variant $V_i$ is independently sampled with replacement.
3. The outcome $Y$ is independently sampled with replacement.

### Density Estimation Sampler

In this scenario we aim to generate a dataset resulting in non-null effect sizes. We do not wish to impose a restricted form for the distribution and hence rely on model evaluation to select the best estimation technique.

Density Estimation Sampler:

1. Fitting
   1. For each variable $Z$ in $(V_1, ..., V_p, Y_1, ...Y_k)$ with parents $X$, an optimal conditional density $x \mapsto \hat{P}_{Z,n}(x)$ is estimated.

2. Sampling
   1. First $W,C$ are jointly sampled with replacement, hence preserving the structure between these variables.
   2. Each genetic variant $V_i$ is sampled from the estimated density: $w \mapsto \hat{P}_{V_i,n}(w)$
   3. The outcome $Y$ is sampled from the estimated density: $(v, w, c) \mapsto \hat{P}_{Y,n}(v, w, c)$

### Estimators' Evaluation
