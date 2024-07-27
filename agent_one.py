class PaulAtreides:
    def __init__(self, n_states, n_actions, lr, gamma, eps, f_model, seed):
        self.Q = np.zeros((n_states, n_actions))
        self.model = f_model(n_states, n_actions, seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def act(self, s):
        if self._rng.random() < self.eps:
            action = self._rng.choice(self.n_actions)
        else:
            action = np.argmax(self.Q[s])
        return action

    def update(self, s, a, r, s_n, update_model: bool):
        max_next_action = np.argmax(self.Q[s_n])
        td_target = r + self.gamma * self.Q[s_n][max_next_action]
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += self.lr * td_error

        if update_model:
            self.model.add(s, a, r, s_n)

    def dream(self, max_steps, **_):
        for _ in range(max_steps):
            s, a, r, s_n = self.model.sample()
            self.update(s, a, r, s_n, update_model=False)
