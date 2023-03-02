from ray.tune.registry import register_env
from LEO_SAT_envronment import LEOSATEnv
from ray import tune

from ray.rllib.algorithms.a3c import A3CConfig
config = A3CConfig()

from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
def env_creator(env_config):
    return LEOSATEnv()  # return an env instance

register_env("LEO", PettingZooEnv(env_creator))

config = A3CConfig() 
config = config.training(lr=0.01, grad_clip=30.0) 
config = config.resources(num_gpus=0) 
config = config.rollouts(num_rollout_workers=4) 
config = config.environment("LEO") 
print(config.to_dict())  
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()  
algo.train()  