# Reinforcement Learning Assignment 4: Proximal Policy Optimization with CartPole


# how to run the experiment:
- pip install -r requirements.txt
- python experiments.py

# how to inspect the optuna hyerparameter optimization results:
- pip install optuna
- install optuna-dashboard: https://optuna.readthedocs.io/en/stable/tutorial/dashboard.html
- run optuna-dashboard: optuna-dashboard sqlite:///ppo_optuna.sqlite3
- open http://localhost:8080 in your browser

# alternatively, inspect results online:
- Alternatively, you can visit https://optuna.github.io/optuna-dashboard/
- upload the ppo_optuna.sqlite3 file
- inspect the results there