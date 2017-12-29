# Optimizing Chemical Reactions with Deep Reinforcement Learning

![TOC](https://raw.githubusercontent.com/lightingghost/chemopt/master/pics/scheme.png)

Zhenpeng Zhou, Xiaocheng Li, Richard N. Zare

The tensorflow implementation of the [paper](http://pubs.acs.org/doi/full/10.1021/acscentsci.7b00492)

# Abstract


Deep reinforcement learning was employed to optimize chemical reactions. Our model iteratively records the results of a chemical reaction and chooses new experimental conditions to improve the reaction outcome. This model outperformed a state-of-the-art blackbox optimization algorithm by using 71% fewer steps on both simulations and real reactions. Furthermore, we introduced an efficient exploration strategy by drawing the reaction conditions from certain probability distributions, which resulted in an improvement on regret from 0.062 to 0.039 compared with a deterministic policy. Combining the efficient exploration policy with accelerated microdroplet reactions, optimal reaction conditions were determined in 30 min for the four reactions considered, and a better understanding of the factors that control microdroplet reactions was reached. Moreover, our model showed a better performance after training on reactions with similar or even dissimilar underlying mechanisms, which demonstrates its learning ability.

# Getting Started

edit the hyperparameters in `config.json` and execute

```python
python lets_start.py
```


# Implementation references

We wish to thank the authors of the following projects for inspiration.

- [Learning to Learn by Gradient Descent by Gradient Descent](https://github.com/deepmind/learning-to-learn)
