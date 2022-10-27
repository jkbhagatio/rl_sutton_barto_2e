"""Sets up and runs dynamic programming agents and algos."""

from dataclasses import dataclass  # field
from typing import Callable, Literal  # Mapping, Sequence, Annotated, List, Union

# import ipdb
import numpy as np

# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
from numpy import typing as npt
from numpy.random import choice

# from scipy import stats


@dataclass(slots=True)
class Dp:
    """Dynamic programming techniques to improve policies.

    Attributes:
        state_action_trans: Possible transitions from each state-action pair to its
            possible successor states. Each row represents a state, and each column
            represents the possible actions for that state. Each row-column element
            itself can be an array, which would contain the possible multiple
            successor states accessible from that state-action pair.
        state_action_trans_p: Probabilities for the transitions in `state_action_trans`.
        state_action_rewards: Rewards corresponding to the transitions in
            `state_action_trans`.
        state_values: State values given the policy.
        action_values: State-action values given the policy.
        policy_probs: The probability of taking each possible action from a given
            state, according to the policy.
        policy_eval: Policy function for choosing an action from state-action values
            in `policy_evaluation()`.
        policy_improve: Policy function for choosing an action from state-action values
            in `policy_improvement()`.
        gamma: Reward discounting term.
    """

    state_action_trans: npt.NDArray[np.float_]
    state_action_trans_p: npt.NDArray[np.float_]
    state_action_rewards: npt.NDArray[np.float_]
    state_values: npt.NDArray[np.float_] = None
    action_values: npt.NDArray[np.float_] = None
    policy_probs: npt.NDArray[np.float_] = None
    policy_eval: Callable = lambda x: choice(x, p=((x + 1) / np.sum(x + 1)))
    policy_improve: Callable = lambda x: choice(np.argwhere(x == np.max(x)).flatten())
    gamma: float = 1

    def __post_init__(self):
        """Initializes some dependent attributes."""
        if self.state_values is None:
            self.state_values = np.zeros(self.state_action_trans.shape[0])
        if self.action_values is None:
            self.action_values = np.zeros_like(self.state_action_trans)
        if self.policy_probs is None:  # set to equiprobable by default
            self.policy_probs = (
                np.ones_like(self.state_action_trans) / self.state_action_trans.shape[1]
            )

    def policy_evaluation(
        self,
        term_thresh: float = 0.01,
        max_iter_ct: int = 100,
        est_type: Literal["state-value", "optimal-policy"] = "state-value",
        use_log: bool = True,
    ):
        """Evaluates a policy by updating state values via sweeps through state-space.

        Terminates after some max state-space iteration count, or when the maximum
        difference in state values between consecutive state-space sweeps is less
        than some value, `term_thresh`.

        Args:
            term_thresh: If state values between consecutive state-space sweeps are
                less than this value, the evaluation algorithm terminates.
            max_iter_ct: The maximum number of iterations to perform through the entire
                state-space before stopping the evaluation algorithm (if consecutive
                state-values do not converge earlier to some value less than
                `term_thresh`).
            est_type: Policy evaluation can be done for either "state-value"
                estimation, or "optimal-policy" estimation. In the former, state-value
                updates consider the entire possible action space, while in the latter,
                state-value updates consider only the action taken by the policy.
            use_log: If true,
        """
        end_flag = False
        iter_ct = 0
        while not end_flag:
            iter_ct += 1
            delta = 0
            for state in range(len(self.state_values)):
                old_val = self.state_values[state]
                # Compute successor state value term for all possible successor states.
                if est_type == "state-value":
                    successor_state_set = self.state_action_trans[state]
                    ss_term_vals = np.zeros(self.state_action_trans.shape[1])
                    for i, s_s in enumerate(successor_state_set):
                        ss_term_vals[i] = np.sum(
                            self.state_action_trans_p[state, i]
                            * (
                                self.state_action_rewards[state, i]
                                + self.gamma * self.state_values[s_s]
                            )
                        )
                    self.state_values[state] = np.sum(
                        self.policy_probs[state] * ss_term_vals
                    )
                    self.action_values[state] = self.state_action_trans_p[state] * (
                        self.state_action_rewards[state]
                        + self.gamma * self.state_values[successor_state_set]
                    )
                # Compute successor state value term for only successor states
                # reachable given the policy's selected action.
                elif est_type == "optimal-policy":
                    action = np.where(
                        self.policy_eval(self.policy_probs[state])
                        == self.policy_probs[state]
                    )[0]
                    action = choice(action) if action.size > 1 else action
                    self.state_values[state] = np.sum(
                        self.state_action_trans_p[state, action]
                        * (
                            self.state_action_rewards[state, action]
                            + self.gamma
                            * self.state_values[self.state_action_trans[state, action]]
                        )
                    )
                    self.action_values[state, action] = self.state_action_trans_p[
                        state, action
                    ] * (
                        self.state_action_rewards[state, action]
                        + self.gamma
                        * self.state_values[self.state_action_trans[state, action]]
                    )
                delta = np.max(
                    np.abs(np.array((delta, (old_val - self.state_values[state]))))
                )
            if (delta < term_thresh) or (iter_ct == max_iter_ct):
                end_flag = True

    def policy_improvement(self, use_log: bool = True) -> bool:
        """Improves a policy via greedy action selection.

        Args:
            use_log: If True,

        Returns:
            stable: If True, no more policy improvement can occur.
        """
        # ipdb.set_trace()
        stable = True
        for state in range(len(self.state_values)):
            old_action = np.where(
                self.policy_eval(self.action_values[state])
                == (self.action_values[state] / np.sum(self.action_values[state]))
            )[0]
            new_action = np.where(
                self.policy_improve(self.action_values[state])
                == (self.action_values[state] / np.sum(self.action_values[state]))
            )[0]
            if (
                self.action_values[state, old_action]
                == self.action_values[state, new_action]
            ):
                stable = False
                break
        return stable

    def policy_iteration(self, max_iter_ct=100, use_log=True):
        """Runs policy iteration via chain of evaluation and improvement.

        Args:
            max_iter_ct: The maximum number of iterations to perform through the entire
                state-space before stopping the evaluation algorithm.
            use_log: If True,
        """
        # ipdb.set_trace()
        stable = False
        iter_ct = 0
        while (not stable) and (iter_ct < max_iter_ct):
            self.policy_evaluation(est_type="optimal-policy")
            stable = self.policy_improvement()
            iter_ct += 1

    def value_iteration(self, use_log=True):
        pass

    def log(self):
        pass
