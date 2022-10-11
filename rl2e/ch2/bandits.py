"""Sets up and runs bandit tasks."""

from dataclasses import dataclass, field
from typing import Callable, List, Union  # Mapping, Sequence, Annotated

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt
from numpy.random import binomial, choice, normal


@dataclass
class Bandit:
    """An RL Bandit Agent.

    Attributes:
        n_arms: Number of bandit arms.
        n_steps: Number of time steps for which the agent performs an action.
        reward_data: The rewards array [n_steps X n_arms]
        action_policy: The action policy to use. Either "e-greedy" or "gradient".
        e_val: The epsilon value to use with "e-greedy" action policy.
        action_values_est: The action value estimate to use. Either "sample-average" or
            "weighted-average".
        initial_action_value: The initial action-value estimates to use (by default
            set to 0 for each arm).
        alpha: The alpha value to use if using "weighted-average" for action-value
            estimate.
        use_unbiased_stepsize: If True, use unbiased stepsize trick with
            weighted-average action-value estimate.
        use_ucb: If True, use upper-confidence-bound action selection for non-greedy
            actions with e-greedy action policy.
        c_val: The c value to use if using Upper-Confidence-Bound algo (`use_ucb`
            must be True)
        step_win: Window size used for computing avg running reward.
        use_save_run: If True, save outcome after each run.
        use_save_avg_reward: If True, save avg running reward in outcome.
        use_save_optimal_action: If True, save optimal action in outcome.
    """

    n_arms: int = 10
    n_steps: int = 5000
    reward_data: npt.NDArray[np.float64] = field(init=False, repr=False)
    action_policy: str = "e-greedy"
    e_val: float = 0.1
    action_value_est: str = "sample-average"
    initial_action_value: npt.NDArray[np.float64] = field(init=False)
    alpha: float = 0.1
    use_unbiased_stepsize: bool = False
    use_ucb: bool = False
    c_val: float = 2.0
    step_win: int = 10
    use_save_run: bool = True
    use_save_avg_reward: bool = True
    use_save_optimal_action: bool = True
    __outcome: pd.DataFrame = field(init=False, repr=False)  # saved run outcomes
    # Action-taken and reward-received vectors, and action values matrix, for all
    # timesteps for the most recent run.
    __actions: npt.NDArray[np.int_] = field(init=False, repr=False)
    __rewards: npt.NDArray[np.float64] = field(init=False, repr=False)
    __action_values: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Initializes some dependent attributes."""
        columns = list(vars(self).keys())[:-3]
        columns += [
            "reward_data",
            "actions",
            "rewards",
            "action_values",
            "running_avg_reward",
            "pct_optimal_action",
        ]
        self.__outcome = pd.DataFrame(columns=columns)
        self.initial_action_value = np.zeros(self.n_arms)

    def gen_reward_data(
        self,
        reward_dist: Union[Callable, List[Callable]] = normal,
        loc: npt.NDArray[np.float64] = None,
        scale: npt.NDArray[np.float64] = None,
    ):
        """Generates reward data for each arm for each step.

        Args:
            reward_dist: The distribution types from which to draw rewards. Can be a
            single distribution, which is used for all arms, or a list of
            distributions, one for each arm.
            loc: The location parameters for the reward distributions for each arm.
                If None, this is set randomly between (0,1) for each arm.
            scale: The scale parameters for the reward distributions for each arm.
                If None, this is set randomly between (0,1) for each arm.
        """
        if loc is None:
            loc = np.random.random_sample((self.n_arms,))
        if scale is None:
            scale = np.random.random_sample((self.n_arms,))
        if type(reward_dist) is not list:
            reward_dist = [reward_dist] * self.n_arms  # type: ignore
        self.reward_data = np.array(
            [
                reward_dist[arm](loc=loc[arm], scale=scale[arm], size=self.n_steps)
                for arm in range(self.n_arms)
            ]
        ).T

    def run(self):
        """Runs bandit agent on task (reward data)."""
        # Initialize actions, rewards, and action values
        self.__actions = np.zeros((self.n_steps,))
        self.__rewards = np.zeros((self.n_steps,))
        self.__action_values = np.zeros_like(self.reward_data)
        all_actions = np.arange(self.n_arms)  # all possible actions
        q_a = self.initial_action_value.copy()  # action-vals vector (current step)
        n_a = np.zeros(self.n_arms)  # action selected counter vector
        n_a = n_a + 0.001 if self.use_ucb else n_a  # set non-zero for division
        if self.action_policy == "e-greedy":  # greedy action mask
            greedy = binomial(1, (1 - self.e_val), self.n_steps)
        if self.action_policy == "gradient":
            h_val = np.zeros(self.n_arms).astype(float)  # initialize action preference
            action = choice(self.n_arms)  # current action
            reward = 0.0  # current reward
        if self.use_unbiased_stepsize:  # set params if using trick
            beta = sigma = self.alpha

        # For each step: take action and get reward + update action-value estimate.
        for step in range(self.n_steps):
            # Take action:
            # find greedy action, accounting for possible multiple optimal actions
            if self.action_policy == "e-greedy":
                greedy_actions = np.argwhere(q_a == np.max(q_a))
                action = int(greedy_actions[choice(len(greedy_actions))])
                # choose randomly from all other actions
                if not greedy[step]:
                    other_actions = all_actions[all_actions != action]
                    if not self.use_ucb:
                        action = int(other_actions[choice(len(other_actions))])
                    else:
                        ucb_weights = q_a[other_actions] + (
                            self.c_val * np.sqrt(np.log(step + 1) / n_a[other_actions])
                        )
                        ucb_actions = np.argwhere(ucb_weights == np.max(ucb_weights))
                        action = int(other_actions[choice(len(ucb_actions))])
            elif self.action_policy == "gradient":
                preferred_actions = np.argwhere(h_val == np.max(h_val))
                action = int(preferred_actions[choice(len(preferred_actions))])
            # Get reward and update action-value.
            self.__actions[step] = action
            self.__rewards[step] = reward = self.reward_data[step, action]
            if self.action_value_est == "sample-average":
                n_a[action] += 1
                alpha = 1 / n_a[action]
            if self.use_unbiased_stepsize:
                alpha = beta / sigma
                sigma += beta * (1 - sigma)
            if self.action_policy == "e-greedy":
                q_a[action] += alpha * (reward - q_a[action])
                self.__action_values[step, :] = q_a
            elif self.action_policy == "gradient":
                policy = np.exp(h_val) / np.sum(np.exp(h_val))  # softmax
                h_val[action] += (
                    alpha
                    * (reward - np.mean(self.__rewards[: (step + 1)]))
                    * (1 - policy[action])
                )
                other_actions = all_actions[all_actions != action]
                h_val[other_actions] -= (
                    alpha
                    * (reward - np.mean(self.__rewards[: (step + 1)]))
                    * policy[other_actions]
                )
                self.__action_values[step, :] = h_val
        if self.use_save_run:
            self.save_run()

    def save_run(self):
        """Saves the outcome from the most recent run."""
        run_num = self.__outcome.shape[0]  # row number
        # add public attributes to df
        for col in self.__outcome.columns:
            if col in dir(self):
                self.__outcome.loc[run_num, col] = getattr(self, col)
        # add hidden attributes to df
        hidden_attribs = list(filter(lambda x: "_Bandit" in x, dir(self)))
        hidden_attribs.remove("_Bandit__outcome")
        for a in hidden_attribs:
            col = a.split("_Bandit__")[-1]
            self.__outcome.loc[run_num, col] = getattr(self, a)
        # map extra outcomes to their flags and functions
        extra_outcome_fn_map = {
            "running_avg_reward": (
                self.use_save_avg_reward,
                self.calc_running_avg_reward,
            ),
            "pct_optimal_action": (
                self.use_save_optimal_action,
                self.calc_optimal_action_pct,
            ),
        }
        # if flag is true, set column value to function output
        for key, val in extra_outcome_fn_map.items():
            if val[0]:
                self.__outcome.loc[run_num, key] = val[1]()

    def get_outcome(self):
        """Returns the saved run outcomes."""
        return self.__outcome

    def calc_running_avg_reward(self):
        """Calculates running average reward from a reward vector and step window."""
        return (
            np.convolve(self.__rewards, np.ones((self.step_win,)), mode="same")
            / self.step_win
        )

    def calc_optimal_action_pct(self):
        """Calculates % optimal action taken at each time step."""
        oa = np.argmax(np.mean(self.reward_data, axis=0))
        oa_mask = self.__actions == oa
        return np.cumsum(oa_mask) / np.arange(
            start=1, stop=(self.reward_data.shape[0] + 1)
        )


def plot_reward_dists(reward_data: Union[np.ndarray, pd.DataFrame]) -> plt.axes:
    """Plots reward distributions given reward data matrix [n_steps X n_arms]."""
    if type(reward_data) is np.ndarray:
        reward_data = pd.DataFrame(reward_data)
    n_arms = reward_data.shape[1]  # number of arms
    ax = sns.violinplot(data=reward_data, scale="count")
    ax = sns.stripplot(
        data=reward_data, jitter=True, zorder=1, palette=["0.7"] * n_arms, ax=ax
    )
    ax.set_title(f"Reward Distributions for each arm of {n_arms}-armed bandit")
    return ax
