- Chapter 1

  - Code up tic-tac-toe and connect4 RL environments + agents. (in relation to questions 1.1, 1.3, 1.4)

  - Code up petroleum refinery operation in 1.2 as both an RL and Control task, see how they differ?

- Chapter 2

  - Code up an E-greedy k-armed bandit algorithm with sample-average value estimation [x]

  - Code up an E-greedy k-armed bandit algorithm with weighted-average value estimation

  - Code up an E-greedy k-armed bandit algorithm with weighted-average value estimation with unbiased step size trick

  - Code up a UCB k-armed bandit algorithm with sample-average OR weighted-average value estimation

  - Code up a gradient k-armed bandit algorithm with sample-average OR weighted-average value estimation


run_agents(r_all=r_all, action_policy=["E-greedy", "UCB", "gradient"], action_value=)

___

def set_bandit_task(K, )

def run_bandit_agents(action_policy=["E-greedy", "UCB", "gradient"], action_value=["sample-average", "weighted-average"]", unbiased_stepsize=["True", "False"])

def plot_reward_dists()

def plot_running_avg_reward()

def plot_optimal_action_pct()


- installable package & instructions
  - dependabot github enable and config
  - use: black, flake8, isort, mypy, pytest, pytest-cov, pre-commit, bandit
    - pyproject.toml (black, flake8, isort)
    - use all in github actions CI
    - add badges for build_env_run_tests and codecov

    