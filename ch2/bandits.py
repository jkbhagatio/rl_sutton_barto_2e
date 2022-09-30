from typing import Callable, Mapping, Sequence, Union, Annotated

import numpy as np
from numpy.random import normal, binomial, choice
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns


def set_bandit_task(k: int = 10, n_steps: int = 5000, reward_dist: Callable = normal, dist_loc: np.array = None,
                    dist_scale: np.array = None):
    """
    Sets up a bandit task by generating rewards for each arm for each step given some parameters.

    Args:
        k: Number of bandit arms.
        n_steps: Number of time steps.
        reward_dist: Distribution from which to draw the rewards.
        dist_loc: The location parameter for the reward distribution. If None, this is set randomly between (0,1) for
            each arm, otherwise this must be a 1-d array with number of elements equal to `k`.
        dist_scale: The scale parameter for the reward distribution. If None, this is set randomly between (0,1) for
            each arm, otherwise this must be a 1-d array with number of elements equal to `k`.

    Returns:
        Rewards array [n_timestamps X n_agents]
    """
    b_means, b_vars = dist_loc, dist_scale
    if b_means is None:
        b_means = reward_dist(loc=0, scale=1, size=k)
    if b_vars is None:
        b_vars = reward_dist(loc=0, scale=1, size=k)
    return np.array([reward_dist(loc=b_means[arm], scale=b_vars[arm], size=n_steps) for arm in range(k)]).T


def run_agent(reward_data: np.array, action_policy: str, action_value: str, alpha: float = None, ucb: bool = False,
              unbiased_stepsize: bool = False, e: float = 0.1, c: float = 2) -> tuple:
    """
    Runs bandit agents for some task given: 1) reward data for each arm for each step; 2) the agent's action policy;
    3) the agent's action values; 4) a flag for using the unbiased stepsize trick for value updates

    Args:
        reward_data: The rewards array [n_steps X n_arms]
        action_policy: The action policy to use. Either "e-greedy" or "gradient"
        action_value: The action value estimate to use. Either "sample-average" or "weighted-average"
        alpha: The alpha value to use if using "weighted-average" for action value.
        ucb: If True, use upper-confidence-bound action selection for non-greedy actions with e-greedy action policy.
        unbiased_stepsize: If True, use unbiased stepesize trick with weighted-average action-value.
        e: The epsilon value to use (only used if "e_greedy" action policy)
        c: The c value to use (only used if "ucb" action policy)

    Returns:
        tuple of two 1-d arrays: reward and action selection at each step
    """
    # Set up.
    n_steps = reward_data.shape[0]
    a_all = np.zeros(n_steps)               # action selected vector
    r_all = np.zeros(n_steps)               # reward vector
    k = reward_data.shape[1]                # number of arms
    all_actions = np.arange(k)              # all possible actions
    q_a = np.zeros(k)                       # action-value estimate vector
    n_a = np.zeros(k)                       # action selected counter vector
    n_a += 0.001 if ucb else n_a
    greedy = binomial(1, (1 - e), n_steps) if action_policy == "e-greedy" else None  # greedy action taken
    if action_policy == "gradient":
        h = np.zeros(k).astype(float)
        policy = np.zeros(k).astype(float)
        a = choice(k)
        r = 0
    if unbiased_stepsize:  # set params if using trick
        sigma = 0
        beta = alpha

    # Action selection, reward, optimal action check, update action-value estimate and action selected counter.
    for step in range(n_steps):
        # action selection
        if action_policy == "e-greedy":
            # find greedy action, accounting for possible multiple optimal actions
            greedy_actions = np.argwhere(q_a == np.max(q_a))
            a = int(greedy_actions[choice(len(greedy_actions))])
            # choose randomly from all other actions
            if not greedy[step]:
                other_actions = all_actions[all_actions != a]
                if not ucb:
                    a = int(other_actions[choice(len(other_actions))])
                else:
                    ucb_weights = q_a[other_actions] + (c * (np.log(step) / n_a[other_actions]))
                    ucb_actions = np.argwhere(ucb_weights == np.max(ucb_weights))
                    a = int(ucb_actions[choice(len(ucb_actions))])
        elif action_policy == "gradient":
            policy = np.exp(h) / np.sum(np.exp(h))
            h[a] += alpha * (r - np.mean(r_all[:step])) * (1 - policy[a])
            other_actions = all_actions[all_actions != a]
            h[other_actions] -= alpha * (r - np.mean(r_all[:step])) * policy[other_actions]

        # Get reward and see if optimal action.
        r_all[step] = r = reward_data[step, a]
        a_all[step] = a
        # Update action-value estimate
        n_a[a] += 1
        alpha = (1 / n_a[a]) if action_value == "sample-average" else alpha
        if unbiased_stepsize:
            alpha = beta / sigma
            sigma += beta * (1 - sigma)
        q_a[a] += alpha * (r - q_a[a])
        return r_all, a_all


def get_running_avg_reward(reward: np.array, step_win: int) -> np.array:
    """Computes running average reward from a reward vector and step window"""
    return np.convolve(reward, np.ones((step_win,)), mode="same") / step_win


def get_optimal_action_pct(reward_data: np.array, action: np.array) -> np.array:
    """Computes % optimal action taken at each time step given reward data matrix
    [n_steps X n_arms] and action vector"""
    oa = np.argmax(np.mean(reward_data, axis=0))
    oa_mask = action == oa
    return np.cumsum(oa_mask) / np.arange(reward_data.shape[0])


def plot_reward_dists(reward_data: Union[np.array, pd.DataFrame]) -> plt.ax:
    """Plots reward distributions and returns axis handle given reward data
     [n_steps X n_arms]"""
    reward_data = pd.DataFrame(reward_data) if type(reward_data) is np.array else reward_data
    k = reward_data.shape[1]  # number of arms
    ax = sns.violinplot(data=reward_data, scale="count")
    ax = sns.stripplot(data=reward_data, jitter=True, zorder=1, palette=["0.7"] * k, ax=ax)
    ax.set_title(f"Reward Distributions for each arm of {K}-armed bandit")
    return ax