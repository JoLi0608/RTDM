import pandas as pd
import wandb
import time
import argparse
import os
# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint",
                    default="/app/data/spinup/sac/continuous_CartPole-v0/cmd_sac_pytorch/cmd_sac_pytorch_s1/")

args = vars(parser.parse_args())
path = args["path"]

wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = path[-2]
wconfig.algo = "SAC"

wconfig.env = path.split("/")[5]

df = pd.read_csv(path+"progress.txt", sep="\t")

data = df.to_dict('records')

for i in data:
    wandb.log(i,step=int(i["TotalEnvInteracts"]),commit=False)


wandb.log({"Done":True},step=int(i["TotalEnvInteracts"]),commit=True)
