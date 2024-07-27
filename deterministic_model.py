class Model:
    def __init__(self, n_states, n_actions, seed):
        self.n_states = n_states
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self.data = []

    def add(self, s: int, a: int, r: float, next_s: int) -> float:
        self.data.append((s, a, r, next_s))
        return r

    def sample(self) -> tuple[int, int, float, int]:
        idx = self._rng.integers(0, len(self.data))
        return self.data[idx]
