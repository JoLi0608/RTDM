import pandas as pd 
import wandb
import matplotlib.pyplot as plt
import math

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import math

# PICKLE
import pickle
infile = open('/app/data/inference_time/data.pkl','rb')
inf_dict = pickle.load(infile)

wandb.init(project="RTDM_percentage", entity="rt_dm")
api = wandb.Api()
env = 'Pusher-v2'
alg = 'ars'
seed = '1'
wconfig = wandb.config
wconfig.algorithm = alg
wconfig.eva_seed = seed
wconfig.env = env
entity, project = "pierthodo", "RTDM_performance"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 
env_step = {'Hopper-v2':0.002,'Pusher-v2':0.01,'continuous_CartPole-v0':0.02,'HalfCheetah-v2':0.01,'Humanoid-v2':0.003}
algs = ['ars','sac','ppo','rtrl','mbpo','pets']
envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2','Humanoid-v2']
env_step = {'Hopper-v2':0.002,'Pusher-v2':0.01,'continuous':0.02,'HalfCheetah-v2':0.01,'Humanoid-v2':0.003}
# envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2']
algs = ['ars','sac','ppo','rtrl','mbpo','pets']
# algs = ['sac','ppo','mbpo','rtrl']
# algs = ['ars','sac','ppo','mbpo','rtrl']
cpus = ['0','0.05','0.1','0.5','1']
gpu = ['1','0']

def testenv(cpus, gpu, alg, env, dic = inf_dict):
    res = {}
    for exp, data in dic.items():
        # print(exp,data)
        ave_inf = sum(data)/len(data)
        exp = exp.split("_")
        # print((exp[1] == cpu), (exp[2] == alg and exp[-1] == gpu))
        if exp[1] in cpus and exp[2] == alg and exp[-1] == gpu and exp[3] == env:
            cpu = exp[1]
            res[cpu] = ave_inf
    # if alg == 'pets':
    #     res['Humanoid-v2'] = 0
    return res 

def cal_repeat(env,gpu = '0'):
    result = []
    for alg in algs:
        # for cpu in cpus:
        dicenv = testenv(cpus, gpu, alg, env)
        result.append(dicenv)
    n = len(result)
    for i in range(n):
        repeat = []
        dicenv = result[i]
        print(i)
        print(dicenv,type(dicenv))
        key = list(dicenv.keys())
        value = list(dicenv.values())
        m = len (dicenv)
        for j in range (m):
            keycpu = cpus[j]
            for k in range (m):
                if key[k] == keycpu:
                    repeat.append((math.floor(value[k]/env_step[env])))
    return repeat

repeat = (env)

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    # for k,v in run.config.items():
    #     if k =='env' and v == 'Hopper-v2':
    #         config_list.append({k: v})
    # print(run.config.items())
    flag = 0
    configs = list( run.config.items())
    wconfig = wandb.config
    wconfig = run.config.items()
    if len(configs) and configs[0][1] == env and configs[2][1] == seed  and configs[3][1] == alg:
        print(configs)
    # and configs[3][1] == 'sac':
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

    # .name is the human-readable name of the run.
        name_list.append(run.name)
        summary_list.append(run.summary._json_dict)
        logs = run.history()
        if len(logs): 
            # print(logs)
            # repeat = list(logs['Action Repeated'])
            # percentage = list(logs['Percentage of Reward Decreased'])
            steps = list(logs['Action Repeated'])
            percents = list(logs['Percentage of Reward Decreased'])
            rewards = list(logs['Reward'])
            # print(rewards,len(rewards))
            n = len(steps)
            initial = rewards[0]
            final= rewards[-1]
            reward_range = initial-final
            for i in range (n):

                delay = env_step[env]*i
                percent1 = percents[i]*100
                print(i,delay, percents[i])
                # print(final,rewards[-1])
                percent = (rewards[i]/initial)*100
                # print(rewards[i])
                # if delay > 0.1:
                #     break
                wandb.log({"Percentage of Reward Decreased": percent, "Percentage of Reward Decreased1": percent1,"Delay": delay,"Reward":rewards[i],'Environment Step Time':env_step[env]},step=i)

            alg = [configs[3][1]] * n
            percent_df = logs[['Action Repeated','Percentage of Reward Decreased']]
            percent_df.insert(2,'Algorithm',alg)
            flag = 1
            
        # print(configs[3])
        break
# # print(config_list)
# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })
# print(runs_df)


# runs_df.to_csv("project.csv")





