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

- installable package & instructions [x]
- GitHub security and analysis checks: dependabot, CodeQL code scanning [x]
- python dev config packages use: black, flake8, mypy, bandit, pytest, pytest-cov, isort, pre-commit
- pyproject.toml (black, flake8, mypy, bandit, pytest, isort)
- github actions CI: (flake8, bandit, mypy, pytest --cov, codecov, )
- add badges for build_env_run_tests and codecov

    