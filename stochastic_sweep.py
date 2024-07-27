def main():

    wandb.init(project="Dune-Q")

    Arrakis = gym.make("Taxi-v3")
    Arrakis = SandStormArrakis(Arrakis)

    # Проверка пространства наблюдений
    n_states = Arrakis.observation_space.n
    n_actions = Arrakis.action_space.n

    print(f"Observation space: {n_states}, Action space: {n_actions}")

    # Инициализация агента
    agent = Muaddib(
        n_states,
        n_actions,
        f_model=StochasticModel,
        lr=0.05,
        gamma=0.975,
        eps=0.1,
        seed=42,
    )

    # Обучения агента с текущими параметрами
    avg_returns = train(
        Arrakis,
        agent,
        on_model_updates=0,
        seed=42,
        n_episodes=3001,
    )

# Define the sweep configuration
sweep_config = {
    "method": "grid",
    "parameters": {
        "n_planning_steps": {"values": [0, 5, 10]},
        "n_dreaming_steps": {"values": [1, 3, 5]}  # Add dreaming steps parameter
    }
}

# Start the sweep
MuaddibTravel = wandb.sweep(sweep=sweep_config, project="Dune-Q")

# Run the sweep
wandb.agent(MuaddibTravel, function=main, count=5)
