class Muaddib:
    def __init__(self, n_states, n_actions, lr, gamma, eps, f_model, seed):
        self.Q = np.zeros((n_states, n_actions))
        self.model = f_model(n_states, n_actions, seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.n_actions = n_actions
        self.n_states = n_states
        self._rng = np.random.default_rng(seed)

    def act(self, s):
        if s >= self.n_states:
            raise ValueError(f"State index {s} is out of bounds for Q-table with size {self.n_states}")

        if self._rng.random() < self.eps:
            action = self._rng.choice(self.n_actions)
        else:
            action_values = self.Q[s]
            max_value = np.max(action_values)
            max_indices = np.where(action_values == max_value)[0]
            action = self._rng.choice(max_indices)
        return action

    def update(self, s, a, r, s_n, update_model: bool):
        if s >= self.n_states or s_n >= self.n_states:
            raise ValueError(f"State index {s} or next state index {s_n} is out of bounds for Q-table with size {self.n_states}")

        max_next_action = np.argmax(self.Q[s_n])
        td_target = r + self.gamma * self.Q[s_n][max_next_action]
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += self.lr * td_error

        if update_model:
            self.model.add(s, a, s_n, r)

    def dream(self, max_steps, **_):
        for _ in range(max_steps):
            s, a = self._rng.choice(self.n_states), self._rng.choice(self.n_actions)
            s_n, reward = self.model.sample_next_state_reward(s, a)
            self.update(s, a, reward, s_n, update_model=False)
