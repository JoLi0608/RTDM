import pandas as pd
import wandb
import time
import argparse
import os
# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint",
                    default="/app/data/pets/HalfCheetah-v2/102236")


args = vars(parser.parse_args())


path = args["path"]



wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = float(os.path.basename(os.path.normpath(path)))
wconfig.algo = "PETS"

dic_name = ["HalfCheetah-v2","Hopper-v2","InvertedPendulum-v2","Pusher-v2","continuous_CartPole-v0"]

for i in dic_name:
    if i in path:
        wconfig.env = i


df = pd.read_csv(path+"results.csv")
data = df.to_dict('records')

for i in data:
    wandb.log(i,step=int(i["env_step"]),commit=False)


wandb.log({"Done":True},step=int(i["env_step"]),commit=True)
