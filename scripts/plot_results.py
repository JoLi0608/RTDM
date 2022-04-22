import pandas as pd
import wandb
import time
import argparse
import os
# Input arguments from command line.


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path",  help="Filepath to trained checkpoint")
parser.add_argument("--algo",  help="Filepath to trained checkpoint")


def load_planet(path="/app/data/planet/default/dmcontrol_walker_walk/2022.04.11/085157/"):
    return {"env":path.split("/")[5],"seed":float(path.split("/")[7]),"algo":"PLANET",
            "data":pd.read_csv(path + "results.csv").to_dict('records')}


args = vars(parser.parse_args())
if args["algo"] == "planet":
    data = load_planet(args["path"])


wandb.init(project="RTDM_train", entity="pierthodo")
wconfig = wandb.config
wconfig.seed = data["seed"]
wconfig.algo = data["algo"]
wconfig.env = data["env"]

for i in data["data"]:
    if data["env"] == "Humanoid-v2" and i["env_step"] > 200000:
        break
    wandb.log(i,step=int(i["env_step"]),commit=False)


wandb.log({"Done":True},commit=True)

