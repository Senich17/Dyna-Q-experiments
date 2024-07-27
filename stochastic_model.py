class StochasticModel:
    def __init__(self, n_states, n_actions, seed):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transitions = np.zeros((n_states, n_actions, n_states))  # Матрица переходов
        self.rewards = np.zeros((n_states, n_actions, n_states))  # Матрица вознаграждений
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def add(self, s, a, next_s, reward):
        # Обновляем матрицы переходов и вознаграждений
        self.transitions[s][a][next_s] += 1
        self.rewards[s][a][next_s] += reward

    def sample(self, s, a):
        # Получаем вероятностное распределение переходов и вознаграждений для данного состояния и действия
        transition_probs = self.transitions[s][a] / np.sum(self.transitions[s][a])  # Вероятности переходов
        reward_probs = self.rewards[s][a] / (np.sum(self.transitions[s][a]) or 1)  # Вероятности вознаграждений (нормализованные по сумме переходов)
        return transition_probs, reward_probs

    def sample_next_state_reward(self, s, a):
        # Выбираем следующее состояние и вознаграждение на основе вероятностного распределения
        transition_probs, reward_probs = self.sample(s, a)
        next_s = self._rng.choice(self.n_states, p=transition_probs)
        reward = self._rng.choice(self.n_states, p=reward_probs[next_s])
        return next_s, reward
