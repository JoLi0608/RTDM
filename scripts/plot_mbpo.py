import pandas as pd
import wandb
import time
import argparse
import os
# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint",
                    default="/app/data/mbpo/default/humanoid_truncated_obs/2022.04.02/170035/")


args = vars(parser.parse_args())


path = args["path"]

wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = float(os.path.basename(os.path.normpath(path)))
wconfig.algo = "MBPO"

dic_name = {"cartpole_continuous":"CartPole-v0",
 "humanoid_truncated_obs":"Humanoid-v2",
 "Hopper-v2":"Hopper-v2",
 "HalfCheetah-v2":"HalfCheetah-v2",
 "pusher":"Pusher-v2"}

for i in dic_name.keys():
    if i in path:
        wconfig.env = dic_name[i]


df = pd.read_csv(path+"results.csv")
data = df.to_dict('records')

for i in data:
    if wconfig.env == "Humanoid-v2" and i["env_step"] > 200000:
        break
    wandb.log(i,step=int(i["env_step"]),commit=False)


wandb.log({"Done":True},commit=True)


