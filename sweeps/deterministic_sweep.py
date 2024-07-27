def main():
    wandb.init(project="Dune-Q")

    n_states = Caladan.observation_space.n
    n_actions = Caladan.action_space.n

    print(f"Observation space: {n_states}, Action space: {n_actions}")

    # Генерируем случайное значение seed на основе текущего времени
    seed = int(time.time())

    agent = PaulAtreides(
        n_states,
        n_actions,
        f_model=Model,
        lr=0.05,
        gamma=0.975,
        eps=0.1,
        seed=seed
    )

    avg_returns = train(
        Caladan,
        agent,
        on_model_updates=0,
        seed=seed,
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
PaulTravel = wandb.sweep(sweep=sweep_config, project="Dune-Q I")

# Run the sweep
wandb.agent(PaulTravel, function=main, count=5)
