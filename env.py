import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

class MaterialEnv(gym.Env):
    def __init__(self):
        super(MaterialEnv, self).__init__()
        self.grid_size = 6  # Define the grid size
        self.materials = ['Cotton', 'Polyester', 'Wool', 'Nylon']
        self.materials_pos = {
            'Cotton': [5, 5],
            'Polyester': [0, 5],
            'Wool': [5, 0],
            'Nylon': [3, 3]
        }
        self.action_space = spaces.Discrete(len(self.materials))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        # Sustainability scores and emojis
        self.sustainability_scores = {
            'Cotton': 1.0,
            'Polyester': 0.2,
            'Wool': 0.6,
            'Nylon': 0.3
        }
        self.material_emojis = {
            'Cotton': '🌱',
            'Polyester': '🧥',
            'Wool': '🧶',
            'Nylon': '🧵'
        }
        
        self.max_steps = 10
        self.current_step = 0
        self.agent_pos = [0, 0]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.agent_pos = [0, 0]
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Set the position of the agent
        obs[tuple(self.agent_pos)] = 1
        return obs

    def step(self, action):
        material = self.materials[action]
        reward = self.sustainability_scores[material]
        done = False
        self.current_step += 1
        
        # Move agent to the new material
        self.agent_pos = self.materials_pos[material]

        if self.current_step >= self.max_steps:
            done = True

        if material == 'Cotton':
            reward = 10
        elif material in ['Polyester', 'Nylon']:
            reward = -10
            done = True
        elif material == 'Wool':
            reward = -5
        else:
            reward = -1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[tuple(self.agent_pos)] = '🤖'  # Emoji for agent
        for mat, pos in self.materials_pos.items():
            grid[tuple(pos)] = self.material_emojis[mat]
        for row in grid:
            print(' '.join(row))
        print(f"Step: {self.current_step}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

