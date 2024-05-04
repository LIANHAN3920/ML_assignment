import numpy as np
import matplotlib.pyplot as plt

def Likelihood(goal_proportion, outcome):
    if outcome == 0:
        likelihood = 1 - goal_proportion
    else:
        likelihood = goal_proportion
    return likelihood


def Bayesian_update(prior_belief, goal_proportions, outcome):

    new_belief = prior_belief

    for i in range(len(goal_proportions)):
        P_proportion = prior_belief[i]
        likelihood = Likelihood(goal_proportions[i], outcome)
        new_belief[i] = P_proportion * likelihood

    new_belief /= np.sum(new_belief)

    return new_belief



true_goal_proportion = 0.6
match_num = 10
match_outcomes = np.random.binomial(1, true_goal_proportion, match_num)

goal_proportions = np.linspace(0, 1, 10)
prior_belief = np.ones_like(goal_proportions)
prior_belief = prior_belief/10

posteriors = []

plt.figure(figsize=(10, 6))
plt.plot(goal_proportions, prior_belief, label='Initial Belief', linestyle='--')

for i in range(1, len(match_outcomes) + 1):
    prior_belief = Bayesian_update(prior_belief, goal_proportions, match_outcomes[i-1])
    l = []
    for j in prior_belief:
        l.append(j)
    posteriors.append(l)

# Next Step: Visualize the analysis

for i, posterior in enumerate(posteriors):
    plt.plot(goal_proportions, posterior, label=f'Posterior after {i+1} match', alpha=0.7)

plt.xlabel('Proportion of Goals Scored')
plt.ylabel('Probability Density')
plt.title('Bayesian Analysis: Estimating Proportion of Goals Scored by a Football Team')
plt.legend()
plt.grid(True)
plt.show()