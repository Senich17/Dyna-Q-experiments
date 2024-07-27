class SandStormArrakis(gym.Wrapper):
    def __init__(self, env, wind_prob=0.2, wind_strength=1):
        super().__init__(env)
        self.wind_prob = wind_prob  # Вероятность воздействия ветра
        self.wind_strength = wind_strength  # Сила воздействия ветра

    def step(self, action):
        # делаем шаг в оригинальной среде
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Определяем, будет ли воздействие ветра на текущем шаге
        if np.random.rand() < self.wind_prob:
            # применим ветер к среде
            wind_action = np.random.choice([-self.wind_strength, 0, self.wind_strength])
            next_state = (next_state + wind_action) % self.env.observation_space.n

        return next_state, reward, terminated, truncated, info
