import gymnasium as gym


class CartPoleEnv(gym.Env):
    """Generates wrapper class for CartPole Env from gymnasium
    
    PARAMS: 
    render_mode -> human or rgb_array
    verbose -> verbosity of environment
    seed -> seed for reproducability

    FUNCTIONS:
    step(a -> action): returns observation, reward, terminated, truncated after action a. If truncated/terminated -> resets env
    terminate(): terminates env


    """
    # SRC: https://arxiv.org/pdf/2407.17032
    def __init__(self, render_mode="rgb_array",verbose= True,seed = 42):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.verbose = verbose
        self.seed = seed
        if verbose:
            print(f"Observation Space={self.env.observation_space}")
            print(f"Action Space={self.env.action_space}")

        # added seed 
        self.observation, self.info = self.env.reset(seed=self.seed)

    def step(self,a):
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(a)
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    

        
    def terminate(self):
        print("Environment terminated")
        self.env.close()
    def reset(self, seed=None):
        self.observation, self.info = self.env.reset()
        return self.observation, self.info