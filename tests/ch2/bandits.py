"""Tests for `bandits.py`."""


# set bandit task:
# `n_arms`: 10, 5
# `n_steps`: 5000, 1000
# `reward_dist` 'normal', 'poisson'
# `dist_loc`: None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# `dist_scale`: None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

reload(bandits)
r_all2, a_all2, q_all2 = bandits.run_bandit_agent(
    reward_data=reward_data,
    action_policy="gradient",
    action_value="weighted-average",
    unbiased_stepsize=True,
)
oa_pct2 = bandits.get_optimal_action_pct(reward_data, a_all2)
oa_pct2

# ___

# e_greedy
  # e_val
  # sample_average
  # weighted_average
    # alpha
    # unbiased_stepsize
  # ucb
    # c_val

# gradient
  # sample_average
  # weighted_average
    # alpha
    # unbiased_stepsize

# defaults
# action_policy = "e-greedy"
# e_vals = (0.1, 0.01, 0.05, 0.2)
# ucb = True, False
# c = (1, 2, 4)
#
# action_value="weighted-average"
# action_value="weighted-average", alpha=0.2
# action_value="weighted-average", alpha=0.05
# action_value="weighted-average", ucb=True, c=1
# action_value="weighted-average", ucb=True, c=4
# action_value="weighted-average", ucb=True, c=1, unbiased_stepsize=True
# action_value="weighted-average", ucb=True, c=4, unbiased_stepsize=True

p0 = [[1, 2], [1, 3], [1, 4], [1, 5]]  # mu and sigma, four versions
p1 = [1, 2, 3]  # values, in this case 3 of them

from itertools import product

matrix_list = [sum(p0) * p1 for p0, p1 in product(p0, p1)]
matrix = zip(*[iter(matrix_list)] * len(p1))
