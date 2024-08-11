import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from env import MaterialEnv

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   nb_actions=actions, nb_steps_warmup=10, 
                   target_model_update=1e-2)
    return dqn

def play_agent():
    env = MaterialEnv()
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_material_env_weights.h5f')

    for episode in range(5):
        obs = env.reset()
        done = False
        while not done:
            action = dqn.forward(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Episode: {episode + 1}, Action: {action}, Reward: {reward}")

if __name__ == "__main__":
    play_agent()

