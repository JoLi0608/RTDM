# import wandb
# api = wandb.Api()
# run = api.run("pierthodo/RTDM_performance/mehq2vmg")
# print(run.history())
# print(run.config)

import pandas as pd 
import wandb
import matplotlib.pyplot as plt
import math
wandb.init(project="RTDM_percentage", entity="rt_dm")
api = wandb.Api()
env = 'continuous_CartPole-v0'
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
            final= rewards[n-1]
            reward_range = initial-final
            for i in range (n):

                delay = env_step[env]*i
                percent1 = (1- ((initial-percents[i])/reward_range))*100
                percent = (rewards[i]/initial)*100
                print(percent)
                # print(rewards[i])
                # if delay > 0.1:
                #     break
                wandb.log({"Percentage of Reward Decreased": percent, "Percentage of Reward Decreased1": percent1,"Delay": delay,"Reward":rewards[i]},step=i)

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


# # Import seaborn
# import seaborn as sns
# print (percent_df)
# # Apply the default theme
# sns.set_theme()

# # fmri = sns.load_dataset("fmri")
# plot = sns.relplot(
#     data=percent_df, kind="line",
#     x="Action Repeated", y="Percentage of Reward Decreased",
#     hue='Algorithm', style='Algorithm',
# )
# plt.show()
# fig = plot.get_figure()
# fig.savefig("Users/liwenyu/Desktop/output.png")


# # Create a visualization
# sns.relplot(
#     data=percent_df,
#     x="total_bill", y="tip", col="time",
#     hue="smoker", style="smoker", size="size",
# )



