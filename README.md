# Dyna-Q-experiments

# Implementation and analysis of the Dyna algorithm in a deterministic and stochastic environment

### Deterministic environment:

• Implementation of the Dyna tabular algorithm.

• Dyna testing with a different number of planning steps starting from 0 (which corresponds to regular Q-learning).

• Comparison of results in the form of graphs of total remuneration and training time for each value of planning steps.

### Stochastic environment:

• Modification of the environment so that it becomes stochastic, making sure that there are at least two states with a non-zero probability of transition.

• Adaptation of the Dyna algorithm to account for the stochasticity of the environment.

### Implementation

In a visual version, the experiments are reflected in the laptop
[https://colab.research.google.com/drive/1-kwpRWv1ubuwV98j0D9T8_qllLc50-JU?usp=sharing]

### Visualization

[https://wandb.ai/senich17/Dune-Q%20I?nw=nwusersenich17]

## Conclusion based on the results of the study 

### Dreaming as a strategy

• Like a strategy, dreaming is aimed at long-term planning and generalization of knowledge. This helps the agent develop a deeper understanding of the environment and develop optimal strategies to achieve long-term goals (for example, not scoring points per step, but winning the game as such).After completing several episodes of the game, the agent switches to the dream mode. It uses all the accumulated data to simulate multiple transitions and update Q-values.

• It is usually used during periods of inactivity or after the end of episodes. This allows the agent to devote time to "thinking" and analyzing a large amount of data.

• Ultimately, this allows the agent to identify long-term patterns and strategies that will help him better handle the game as a whole.

### Planning as a tactic

• Like tactics, planning focuses on short-term actions and adaptation to current conditions. This helps the agent to react quickly to changes in the environment and optimize their actions in the short term (a set of points for each step, in a moment).

• Used regularly, for example, after each step or set of steps. Frequent updating of Q-values allows the agent to quickly adjust their actions.

• The goal is to improve current decisions and behavior based on the latest data received.

### The result

• Planning (tactical approach) helps the agent to respond quickly and effectively to current challenges and conditions, optimizing his actions in the short term. The agent plays a game and after each action, for example, takes 5 planning steps. It uses a model to simulate nearby transitions and update Q-values.

• A strategic approach allows the agent to summarize the accumulated experience and develop optimal strategies to achieve long-term goals.

• Using both approaches in Dyna-Q helps to create a balanced agent that can act effectively in both the short and long term.
