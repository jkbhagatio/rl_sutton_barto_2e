"""Sets up and runs bandit tasks."""

from typing import Callable, Union  # Callable, Mapping, Sequence, Union, Annotated

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import binomial, choice, normal


def set_bandit_task(
    n_arms: int = 10,
    n_steps: int = 5000,
    reward_dist: Callable = normal,
    dist_loc: np.ndarray = None,
    dist_scale: np.ndarray = None,
) -> np.ndarray:
    """Sets up a bandit task by generating reward for each arm for each step.

    Args:
        n_arms: Number of bandit arms.
        n_steps: Number of time steps.
        reward_dist: Distribution from which to draw the rewards.
        dist_loc: The location parameter for the reward distribution. If None, this is
            set randomly between (0,1) for each arm, otherwise this must be a 1-d array
            with number of elements equal to `k`.
        dist_scale: The scale parameter for the reward distribution. If None, this is
            set randomly between (0,1) for each arm, otherwise this must be a 1-d array
            with number of elements equal to `k`.

    Returns:
        Rewards array [n_timestamps X n_agents]
    """
    b_means, b_vars = dist_loc, dist_scale
    if b_means is None:
        b_means = np.random.random_sample((10,))
    if b_vars is None:
        b_vars = np.random.random_sample((10,))
    return np.array(
        [
            reward_dist(loc=b_means[arm], scale=b_vars[arm], size=n_steps)
            for arm in range(n_arms)
        ]
    ).T


def run_bandit_agent(
    reward_data: np.ndarray,
    action_policy: str = "e-greedy",
    e_val: float = 0.1,
    action_value: str = "sample-average",
    initial_values: np.ndarray = None,
    alpha: float = 0.1,
    unbiased_stepsize: bool = False,
    ucb: bool = False,
    c_val: float = 2,
) -> tuple:
    """Runs action bandit agent specced by some parameters, given reward data.

    Args:
        reward_data: The rewards array [n_steps X n_arms] (see `set_bandit_task()`)
        action_policy: The action policy to use. Either "e-greedy" or "gradient"
        e_val: The epsilon value to use (only used if "e-greedy" action policy)
        action_value: The action value estimate to use. Either "sample-average" or
            "weighted-average"
        initial_values: The initial action-values (if None, then set to 0 for each arm)
        alpha: The alpha value to use if using "weighted-average" for action value.
        unbiased_stepsize: If True, use unbiased stepsize trick with weighted-average
            action-value.
        ucb: If True, use upper-confidence-bound action selection for non-greedy actions
            with e-greedy action policy.
        c_val: The c_val value to use (only used if "ucb" is True)

    Returns:
        tuple of two 1-d arrays: reward and action selection at each step
    """
    # Set up.
    n_steps = reward_data.shape[0]
    a_all = np.zeros(n_steps)  # action selected vector
    r_all = np.zeros(n_steps)  # reward vector
    n_arms = reward_data.shape[1]  # number of arms
    all_actions = np.arange(n_arms)  # all possible actions
    q_a = np.zeros(n_arms) if initial_values is None else initial_values  # action-vals
    q_a_all = np.zeros_like(reward_data)
    n_a = np.zeros(n_arms)  # action selected counter vector
    n_a += 0.001 if ucb else n_a  # set to small non-zero val for division
    if action_policy == "e-greedy":  # greedy action mask
        greedy = binomial(1, (1 - e_val), n_steps)
    if action_policy == "gradient":
        h_val = np.zeros(n_arms).astype(float)
        action = choice(n_arms)  # current action
        reward = 0.0  # current reward
    if unbiased_stepsize:  # set params if using trick
        beta = sigma = alpha

    # For each step: take action and get reward + update action-value estimate.
    for step in range(n_steps):
        # Take action.
        if action_policy == "e-greedy":
            # find greedy action, accounting for possible multiple optimal actions
            greedy_actions = np.argwhere(q_a == np.max(q_a))
            action = int(greedy_actions[choice(len(greedy_actions))])
            # choose randomly from all other actions
            if not greedy[step]:
                other_actions = all_actions[all_actions != action]
                if not ucb:
                    action = int(other_actions[choice(len(other_actions))])
                else:
                    ucb_weights = q_a[other_actions] + (
                        c_val * (np.log(step) / n_a[other_actions])
                    )
                    ucb_actions = np.argwhere(ucb_weights == np.max(ucb_weights))
                    action = int(ucb_actions[choice(len(ucb_actions))])
        elif action_policy == "gradient":
            preferred_actions = np.argwhere(h_val == np.max(h_val))
            action = int(preferred_actions[choice(len(preferred_actions))])
        # Get reward and update action-value.
        a_all[step] = action
        r_all[step] = reward = reward_data[step, action]
        if action_value == "sample-average":
            n_a[action] += 1
            alpha = 1 / n_a[action]
        if unbiased_stepsize:
            alpha = beta / sigma
            sigma += beta * (1 - sigma)
        if action_policy == "e-greedy":
            q_a[action] += alpha * (reward - q_a[action])
            q_a_all[step, action] = q_a[action]
        elif action_policy == "gradient":
            policy = np.exp(h_val) / np.sum(np.exp(h_val))
            h_val[action] += (
                alpha * (reward - np.mean(r_all[: (step + 1)])) * (1 - policy[action])
            )
            other_actions = all_actions[all_actions != action]
            h_val[other_actions] -= (
                alpha * (reward - np.mean(r_all[: (step + 1)])) * policy[other_actions]
            )
            q_a_all[step, :] = h_val

    return r_all, a_all, q_a_all


def get_running_avg_reward(reward: np.ndarray, step_win: int) -> np.ndarray:
    """Computes running average reward from a reward vector and step window."""
    return np.convolve(reward, np.ones((step_win,)), mode="same") / step_win


def get_optimal_action_pct(reward_data: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Computes % optimal action taken at each time step.

    From a reward data matrix [n_steps X n_arms] and action selected vector [n_steps].
    """
    oa = np.argmax(np.mean(reward_data, axis=0))
    oa_mask = action == oa
    return np.cumsum(oa_mask) / np.arange(start=1, stop=(reward_data.shape[0] + 1))


def plot_reward_dists(reward_data: Union[np.ndarray, pd.DataFrame]) -> plt.axes:
    """Plots reward distributions given reward data matrix [n_steps X n_arms]."""
    reward_data = (
        pd.DataFrame(reward_data) if type(reward_data) is np.ndarray else reward_data
    )
    n_arms = reward_data.shape[1]  # number of arms
    ax = sns.violinplot(data=reward_data, scale="count")
    ax = sns.stripplot(
        data=reward_data, jitter=True, zorder=1, palette=["0.7"] * n_arms, ax=ax
    )
    ax.set_title(f"Reward Distributions for each arm of {n_arms}-armed bandit")
    return ax
