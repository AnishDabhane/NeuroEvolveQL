import numpy as np
from environment import env
from agent import DQNagent


input_values = ['here goes the input values']

state = np.array(input_values)

rewards_Arr = ["again rewards array"]


def train(agent, num_generation, env):
    for generation in range(num_generation):
        total_reward = 0
        done = False
        state = env.current_state
        print("generation:", generation)
        for i in range(10):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.update_q_values()
            state = next_state
            total_reward += reward
            print("Action:", action, "Reward:", reward, "Total Reward:", total_reward)
            if done:
                break



agent = DQNagent(num_states=10, num_actions=10)
environment = env(state)
train(agent, num_generation=10, env=environment)

new_states=np.array(["input array"])
pred=agent.predict(new_states)


print(pred)