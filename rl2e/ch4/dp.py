"""Sets up and runs dynamic programming agents and algos."""

from dataclasses import dataclass  # field
from typing import Callable  # Literal, Mapping, Sequence, Annotated, List, Union

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
        action_trans: Possible transitions from each state-action pair to its
            possible successor states. Each row represents a state, and each column
            represents the possible actions for that state. Each row-column element
            itself can be an array, which would contain the possible multiple
            successor states accessible from that state-action pair.
        action_trans_p: Probabilities for the transitions in `action_trans`.
        action_rewards: Rewards corresponding to the transitions in
            `action_trans`.
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

    action_trans: npt.NDArray[np.float_]
    action_trans_p: npt.NDArray[np.float_]
    action_rewards: npt.NDArray[np.float_]
    state_values: npt.NDArray[np.float_] = None
    action_values: npt.NDArray[np.float_] = None
    policy_probs: npt.NDArray[np.float_] = None
    # These policies will find the *action-value* of the policy, we then have to
    # reverse-index from this value to find the actual action.
    # By default, `policy_eval` will choose an action from a state weighted by
    # the relative action-values for all actions from that state.
    policy_eval: Callable = lambda x: choice(x, p=((np.exp(x)) / np.sum(np.exp(x))))
    policy_improve: Callable = lambda x: np.max(x)
    gamma: float = 1

    def __post_init__(self):
        """Initializes some dependent attributes."""
        if self.state_values is None:
            self.state_values = np.zeros(self.action_trans.shape[0])
        if self.action_values is None:
            self.action_values = np.zeros_like(self.action_trans)
        if self.policy_probs is None:  # set to equiprobable by default
            self.policy_probs = (
                np.ones_like(self.action_trans) / self.action_trans.shape[1]
            )

    def policy_evaluation(
        self,
        term_thresh: float = 0.001,
        max_iter_ct: int = 100,
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
                successor_state_set = self.action_trans[state]
                ss_term_vals = np.zeros(self.action_trans.shape[1])
                for i, s_s in enumerate(successor_state_set):
                    ss_term_vals[i] = np.sum(
                        self.action_trans_p[state, i]
                        * (
                            self.action_rewards[state, i]
                            + self.gamma * self.state_values[s_s]
                        )
                    )
                self.state_values[state] = np.sum(
                    self.policy_probs[state] * ss_term_vals
                )
                self.action_values[state] = self.action_trans_p[state] * (
                    self.action_rewards[state]
                    + self.gamma * self.state_values[successor_state_set]
                )
                delta = np.max(
                    np.abs(np.array((delta, (old_val - self.state_values[state]))))
                )
            if (delta < term_thresh) or (iter_ct == max_iter_ct):
                end_flag = True
        # ipdb.set_trace()

    def policy_improvement(self, use_log: bool = True) -> bool:
        """Improves a policy via greedy action selection.

        Args:
            use_log: If True,

        Returns:
            stable: If True, no more policy improvement can occur.
        """
        # ipdb.set_trace()
        stable = True
        # Update `policy_probs` based on `action_values`
        a_t, a_v, = (
            self.action_trans,
            self.action_values,
        )
        self.policy_probs = (
            np.exp(a_v)
            / np.tile(np.sum(np.exp(a_v), axis=1), (a_t.shape[1], 1)).transpose()
        )
        # Compute `old_action` and `new_action` from `policy_probs`
        for state in range(len(self.state_values)):
            old_action_val = self.policy_eval(self.policy_probs[state])
            old_action = np.argmin(np.abs(self.policy_probs[state] - old_action_val))
            new_action_val = self.policy_improve(self.policy_probs[state])
            new_action = np.argmin(np.abs(self.policy_probs[state] - new_action_val))
            if not old_action == new_action:
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
            self.policy_evaluation()
            stable = self.policy_improvement()
            iter_ct += 1

    def value_iteration(
        self,
        term_thresh: float = 0.01,
        max_iter_ct: int = 100,
        use_log: bool = True,
    ):
        pass

    def log(self):
        pass
