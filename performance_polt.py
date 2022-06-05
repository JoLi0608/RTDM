import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from sklearn.linear_model import LinearRegression
sns.set_style('whitegrid')
from scipy.signal import lfiltic, lfilter
import pandas as pd 
import matplotlib
matplotlib.rc('font', **font)

dt_dic = {"Hopper-v2":0.002, "HalfCheetah-v2":0.01,"continuous_CartPole-v0":0.02,"Humanoid-v2":0.003,"Pusher-v2":0.01,
         "dmc_walker_walk":0.0025,"dmc_cheetah_run":0.01}
inference_time = pickle.load(open("inference_time.pkl","rb"))
env_list = dt_dic.keys()
algo_list = ["sac","ppo","ars","mbpo","rtrl"]
# algo_list = ["sac","ppo","ars","mbpo","rtrl", "pets",]

def load_repeat(mean=True):
    data_repeat = pickle.load(open("repeat.pkl","rb"))
    config_list, data = data_repeat
    d_repeat = defaultdict(list)
    for i in range(len(config_list)):
        try:
            d_repeat[config_list[i]["env"]+"_"+config_list[i]["algorithm"]].append(data[i])
        except:
            print("Error on ",i)
    new_dic = {}
    for i in d_repeat.keys():
        try:
            tmp = np.array([n["Reward"] for n in d_repeat[i]]).mean(axis=0)
            new_dic[i]= d_repeat[i][0]
            new_dic[i]["Reward"] = tmp
        except:
            continue
    return new_dic


# x-axis being action repeated and delay 
# y-axis being performance and percentage of performance
# (Figures 4.3 4.4 and B.3 in the thesis)
font = {'family' : 'normal',
        'size'   : 15}
for x_axis in ["Action_Repeat", "Delay"]:
    for y_axis in ["Reward","Reward Percentage"]:
        config,repeat_data = pickle.load(open("repeat.pkl","rb"))
        fig, ax_list = plt.subplots(2, 3,figsize=(14,8))
        ax_list = ax_list.flatten()
        for idx,env_name in enumerate(["Hopper-v2","HalfCheetah-v2","Humanoid-v2","continuous_CartPole-v0","dmc_walker_walk","dmc_cheetah_run"]):
            data = []
            for i in range(len(config)):
                try: # This section is used to format the data into a pandas dataframe 
                    if config[i]["env"] == env_name :
                        repeat_data[i]["Algo"] = config[i]["algorithm"]
                        repeat_data[i]["Env"] = config[i]["env"]
                        tmp = repeat_data[i]
                        if config[i]["algorithm"] == "rtrl": # if the algorithm is rtrl you need to adjust the number of step you use
                            tmp["_step"] = tmp["_step"] * 5 + 5
                            data.append(tmp[:5])
                        else:
                            data.append(tmp[:25])
                except:
                    continue

            df = pd.concat(data)
            df = df.reset_index()
            df = df.drop(columns="index")
            df["Reward Percentage"] = "NaN"
            alg_list = pd.unique(df["Algo"].values.ravel()) # Get the list of algorithm
            tmp_list = ["dreamer","planet"]if idx > 4 else alg_list 
            for algo in tmp_list:
                try:
                    tmp = df[(df["Algo"]==algo) ]["Reward"]
                    max_r = np.max(tmp)
                    min_r = np.min(tmp)
                except:
                    print(df[(df["Algo"]==algo) ])
                df.loc[df["Algo"]==algo, "Reward Percentage"]= 100*(1-((max_r -df[(df["Algo"]==algo) ]["Reward"])/(max_r-min_r)))

            disc = 1 if idx > 3 else 5
            df["Delay"] = df["_step"]*(dt_dic[env_name]/disc)
            df["Action_Repeat"] = df["_step"]
            df["Algo"]= df["Algo"].apply(lambda s: s.upper())
            tmp_algo_list = [i.upper() for i in algo_list]
            if idx >3:
                sns.lineplot(data=df,x=x_axis,y=y_axis,hue="Algo",ax=ax_list[idx], linestyle='--').set_title(env_name)            
            else:
                if env_name == "Humanoid-v2":
                    sns.lineplot(data=df,x=x_axis,y=y_axis,hue="Algo",ax=ax_list[idx],hue_order=tmp_algo_list).set_title(env_name)
                else:
                    if "continuous" in env_name:
                        sns.lineplot(data=df,x=x_axis,y=y_axis,hue="Algo",ax=ax_list[idx],hue_order=tmp_algo_list+["PETS"]).set_title("CartPole-v0")
                    else:
                        sns.lineplot(data=df,x=x_axis,y=y_axis,hue="Algo",ax=ax_list[idx],hue_order=tmp_algo_list+["PETS"]).set_title(env_name)
            if idx == 0:
                ax_list[idx].legend(loc=(3.95,0.2))
            elif idx == 5:
                h = plt.gca().get_lines()
                ax_list[idx].legend(handles=h, labels=["DREAMER","PLANET"], loc=(1.1,0.2))
            else:
                ax_list[idx].legend([],[], frameon=False)
            ax_list[idx].axvline(x=dt_dic[env_name])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=0.4)
        plt.tight_layout()
        plt.savefig(x_axis+y_axis+".pdf", bbox_inches="tight")

# # x-axis being number of CPUs 
# # y-axis being performance and percentage of performance
# # (Figure 4.5 in the thesis)

# font = {'family' : 'normal',
#         'size'   : 13}
# make_s = lambda algo,env,gpu,cpu_num: "cpu_"+str(cpu_num)+"_"+algo+"_"+env+"_gpu_"+str(gpu)
# data_repeat = load_repeat()
# _,ax_list = plt.subplots(2,4,figsize=(15,6))

# matplotlib.rc('font', **font)
# for idx_c,plot_content in enumerate(["Reward Percentage","Reward"]):

#     for idx,env in enumerate(["Hopper-v2","HalfCheetah-v2","Humanoid-v2","continuous_CartPole-v0"]):
#         if "continuous" in env:
#             cpu_num = 0.05
#         else:
#             cpu_num = 0.5
#         calc = lambda x,cpu_time: [cpu_time * (cpu_num/i)**2 for i in x]
#         pd_data = []
#         for algo in algo_list+["pets"]:
#             if env == "Humanoid-v2" and algo == "pets":
#                 continue

#             r_list = []
#             cpu_time = np.median(inference_time[make_s(algo,env,0,cpu_num)])

#             x_t = [i*0.01 + 0.01 for i in range(int(cpu_num*100))]
#             t = calc(x_t,cpu_time)
#             for inf_time in t:
#                 if algo in ["rtrl","pets"]:
#                     dt = dt_dic[env]
#                 else:
#                     dt = dt_dic[env]/5
#                 repeat = int(inf_time/dt)
#                 repeat = 49 if repeat > 49 else repeat
#                 if algo in ["rtrl","pets"]:
#                     repeat = 9 if repeat > 9 else repeat
#                 r_list.append(data_repeat[env+"_"+algo].iloc[repeat]["Reward"])
#             p_list = 100*(1-((r_list[-1]-r_list)/(np.max(r_list)-np.min(r_list))))
#             p_list = [100 if i > 100 else i for i in p_list]
#             if algo == "ars" and env == "continuous_CartPole-v0":
#                 p_list = [100]*len(r_list)
#             if algo == "pets":
#                 p_list = [0]*len(r_list)
#             if plot_content == "Reward Percentage":
#                 ax_list[idx_c,idx].plot(x_t,p_list,label=algo.upper())
#             else:
#                 ax_list[idx_c,idx].plot(x_t,r_list,label=algo.upper())
#         if idx_c == 0:
#             if "continuous" in env:
#                 ax_list[idx_c,idx].set_title("CartPole-v0")
#             else:
#                 ax_list[idx_c,idx].set_title(env)
#         else:
#             ax_list[idx_c,idx].set_xlabel("CPU")
#         ax_list[idx_c,idx].set_ylabel(plot_content)
#         if idx == 0 and idx_c==1:
#             ax_list[idx_c,idx].legend(loc=(0.85,-0.5), ncol=6)
#         if idx != 0:
#             ax_list[idx_c,idx].set_ylabel("")
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
# plt.tight_layout()
# plt.savefig("performance_adjusted.pdf", bbox_inches="tight")
