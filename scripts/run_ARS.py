
import ray
from ray import tune
import mbrl
ray.init()
tune.run(
    "ARS",
    stop={"timesteps_total": 1000000},
    config={
        "env": "continuous_CartPole-v0",
        "num_rollouts": tune.grid_search([4,8,32,64]),
        "lr": tune.grid_search([0.01, 0.02, 0.025]),
        "noise_stdev": tune.grid_search([0.03,0.025,0.02,0.01]),
    },
)



