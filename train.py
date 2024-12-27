import json
from stun import QLearningAgent, SchedulerEnv


# Main function to run Q-learning
def train_q_learning(env, num_episodes=500, max_steps_per_episode=100, output_file="training_data.json"):
    agent = QLearningAgent(env)
    training_data = {
        "rewards_per_episode": [],
        "exploration_rates": [],
        "performance_fps": []
    }

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_exploration()
        training_data["rewards_per_episode"].append(total_reward)
        training_data["exploration_rates"].append(agent.exploration_rate)
        training_data["performance_fps"].append(env.evaluate_performance(env.params))
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.4f}")

    # Save data to JSON file
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"Training data saved to {output_file}")
    return agent

if __name__ == "__main__":
    env = SchedulerEnv(filter_threshold=0.2)
    trained_agent = train_q_learning(env)
