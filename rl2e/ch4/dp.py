"""Sets up and runs dynamic programming agents and algos."""

from dataclasses import dataclass, field
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
    z0: float = 1e-6  # term to avoid division by 0 errors
    policy_eval: Callable = field(init=False)
    policy_improve: Callable = lambda x: choice(np.argwhere(x == np.max(x)).flatten())
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
        # By default, `policy_eval` will choose an action from a state weighted by
        # the relative action-values for all actions from that state.
        self.policy_eval = lambda x: choice(x, p=((x + self.z0) / np.sum(x + self.z0)))

    def policy_evaluation(
        self,
        term_thresh: float = 0.01,
        max_iter_ct: int = 100,
        est_type: Literal["eval", "iter"] = "eval",
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
            est_type: Policy evaluation can be done for either "eval"
                estimation, or "iter" estimation. In the former, we consider
                evaluation purely for evaluation's sake: state-value updates consider
                the entire possible action space; in the latter, we consider
                evaluation for policy iteration: state-value updates consider only
                the action taken by the policy.
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
                if est_type == "eval":
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
                # Compute successor state value term for only successor states
                # reachable given the policy's selected action.
                elif est_type == "iter":
                    action_val = self.policy_eval(self.policy_probs[state])
                    action = np.argwhere(action_val == self.policy_probs[state])
                    action = choice(action.flatten())
                    self.state_values[state] = np.sum(
                        self.action_trans_p[state, action]
                        * (
                            self.action_rewards[state, action]
                            + self.gamma
                            * self.state_values[self.action_trans[state, action]]
                        )
                    )
                    self.action_values[state, action] = self.action_trans_p[
                        state, action
                    ] * (
                        self.action_rewards[state, action]
                        + self.gamma
                        * self.state_values[self.action_trans[state, action]]
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
        # Update `policy_probs` based on `action_values`
        s_a_t, a_v, z0 = self.action_trans, self.action_values, self.z0
        self.policy_probs = (a_v + z0) / np.tile(
            np.sum(a_v + z0, axis=1), (s_a_t.shape[1], 1)
        ).transpose()
        # Compute `old_action` and `new_action` from `policy_probs`
        for state in range(len(self.state_values)):
            old_action_val = self.policy_eval(self.policy_probs[state])
            old_action = np.argmin(np.abs(a_v - old_action_val))
            new_action_val = self.policy_improve(self.policy_probs[state])
            new_action = np.argmin(np.abs(a_v - new_action_val))
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
            self.policy_evaluation(est_type="iter")
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
