import rtrl
import argparse
import os
from rtrl import Training, run
from rtrl.wrappers import StatsWrapper

# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint",
                    default="/app/data/rtrl/5/checkpoint/Pusher-v2/04-05-2022-07-10-57/")

args = vars(parser.parse_args())

path = args["path"]
path_split = path.split(path,"/")

#wandb.init(project="RTDM_train", entity="pierthodo")
#wconfig = wandb.config
#wconfig.seed = float(path_split[4])
#wconfig.algo = "RTRL"
#wconfig.env = path_split[6]

r = rtrl.load(path+"store")

env = StatsWrapper(r.Env(seed_val=r.seed+r.epoch), window=r.stats_window or r.steps)
for step in range(self.steps):
    action, training_stats = self.agent.act(*env.transition, train=True)
    stats_training += training_stats
    env.step(action)

print(**env.stats())
