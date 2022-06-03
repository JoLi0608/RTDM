import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import math

import pickle
infile = open('/data/data.pkl','rb')
inf_dict = pickle.load(infile)


# 1. same algorithm same condition, difference of different environment
# envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2','Humanoid-v2']
# # envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2']
# algs = ['ars','sac','ppo','rtrl','mbpo','pets']
# # algs = ['sac','ppo','mbpo','rtrl']
# # algs = ['ars','sac','ppo','mbpo','rtrl']
# cpus = ['5','4','2','1','0.5','0.1','0.05','0.01']
# gpu = ['1','0']

# def testenv(cpu, gpu, alg, envs, dic = inf_dict):
#     res = {}
#     for exp, data in dic.items():
#         # print(exp,data)
#         ave_inf = sum(data)/len(data)
#         exp = exp.split("_")
#         # print((exp[1] == cpu), (exp[2] == alg and exp[-1] == gpu))
#         if exp[1] == cpu and exp[2] == alg and exp[-1] == gpu and exp[3] in envs:
#             env = exp[3]
#             res[env] = ave_inf
#     if alg == 'pets':
#         res['Humanoid-v2'] = 0
#     return res 
# envlabel = ['Hopper','Pusher','CartPole','HalfCheetah','Humanoid']
# result = []
# bar_width = 0.15
# objects = envlabel
# y_pos = np.arange(len(objects))
# for alg in algs:
#     # for cpu in cpus:
#     dicenv = testenv('5', '0', alg, envs)
#     result.append(dicenv)
# n = len(result)
# for i in range(n):
#     inference = []
#     dicenv = result[i]
#     print(i)
#     print(dicenv,type(dicenv))
#     key = list(dicenv.keys())
#     value = list(dicenv.values())
#     m = len (dicenv)
#     for j in range (m):
#         keyenv = envs[j]
#         for k in range (m):
#             if key[k] == keyenv:
#                 inference.append(value[k])
#     plt.yscale('log')
#     # plt.grid(color='k', linestyle='--', linewidth=0.5)
#     ax = plt.bar(y_pos+(i*bar_width), inference, bar_width)
#     plt.axis([ -0.25, 5,1E-6, 1])
#     plt.xticks(range(len(objects)), objects)
#     # ax.set(ylim=(1E-6, 1E-2))

        


# plt.xticks(y_pos, objects)
# plt.legend(algs,loc=1)
# plt.ylabel('Inference Time (seconds)')
# plt.title('Inference Time with 5 CPU 0 GPU')

# plt.show()

# dicenv1 = testenv('1', '1', 'ppo', envs)
# dicenv2 = testenv('1', '1', 'ars', envs)
# objects = dicenv1.keys()
# y_pos = np.arange(len(objects))
# inference1 = dicenv1.values()
# inference2 = dicenv2.values()
# algs = ['PPO', 'ARS']
# bar_width = 0.2
# plt.bar(y_pos, inference1, bar_width, alpha=0.5)
# plt.bar(y_pos+bar_width, inference1, bar_width, alpha=0.5)
# plt.legend(algs,loc=2)
# plt.xticks(y_pos, objects)
# plt.ylabel('Inference Time (seconds)')
# plt.title('inference time in different environments')
# plt.show()


# 2. same environment same condition, difference of different algorithm

# # envs = ['Hopper-v2','Humanoid-v2','Pusher-v2','continuous','HalfCheetah-v2']
# envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2','Humanoid-v2']
# # algs = ['ars','sac','ppo','pets','mbpo','planet','rtrl']
# algs = ['ars', 'sac','ppo','rtrl','mbpo','pets']
# # algs = ['ars','sac','ppo','mbpo','rtrl']
# cpus = ['5','4','2','1','0.5','0.1','0.05','0.01']
# gpu = ['1','0']

# def testenv(cpu, gpu, env, algs, dic = inf_dict):
#     res = {}
#     for exp, data in dic.items():
#         # print(exp,data)
#         ave_inf = sum(data)/len(data)
#         exp = exp.split("_")
#         # print((exp[1] == cpu), (exp[2] == alg and exp[-1] == gpu))
#         if exp[1] == cpu and exp[2] in algs and exp[-1] == gpu and exp[3] == env:
#             alg = exp[2]
#             res[alg] = ave_inf
#     if env == 'Humanoid-v2':
#         res['pets'] = 0
#     return res 

# result = []
# bar_width = 0.15
# objects = algs
# y_pos = np.arange(len(objects))
# for env in envs:
#     # for cpu in cpus:
#     dicenv = testenv('0.05', '0', env, algs)
#     result.append(dicenv)

# n = len(result)
# for i in range(n):
#     inference = []
#     dicenv = result[i]
#     print(i)
#     print(dicenv,type(dicenv))
#     key = list(dicenv.keys())
#     value = list(dicenv.values())
#     m = len (dicenv)
#     for j in range (m):
#         keyalg = algs[j]
#         for k in range (m):
#             if key[k] == keyalg:
#                 inference.append(value[k])
#                 break
#     # objects = dicenv.keys()
#     # y_pos = np.arange(len(objects))
#     plt.yscale('log')
#     # plt.grid(color='k', linestyle='--', linewidth=0.5)
#     ax = plt.bar(y_pos+(i*bar_width), inference, bar_width)
#     plt.axis([ -0.25, 6,1E-6, 10])
#     plt.xticks(range(len(objects)), objects)
#     # ax.set(ylim=(1E-6, 1E-2))

        


# plt.xticks(y_pos, objects)
# plt.legend(envs,loc=2)
# plt.ylabel('Inference Time (seconds)')
# plt.title('Inference Time with 0.05 CPU 0 GPU')


# plt.show()

# 3. same environment same algorithm, difference of different GPU
envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2','Humanoid-v2']
env_step = {'Hopper-v2':0.002,'Pusher-v2':0.01,'continuous':0.02,'HalfCheetah-v2':0.01,'Humanoid-v2':0.003}
# envs = ['Hopper-v2','Pusher-v2','continuous','HalfCheetah-v2']
algs = ['ars','sac','ppo','rtrl','mbpo','pets']
# algs = ['sac','ppo','mbpo','rtrl']
# algs = ['ars','sac','ppo','mbpo','rtrl']
cpus = ['5','1','0.5','0.1','0.05']
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
result = []
bar_width = 0.15
objects = cpus
gpu = '0'
env = 'Hopper-v2'
y_pos = np.arange(len(objects))
for alg in algs:
    # for cpu in cpus:
    dicenv = testenv(cpus, gpu, alg, env)
    result.append(dicenv)
n = len(result)
for i in range(n):
    inference = []
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
                # inference.append(value[k])
                repeat.append((math.floor(value[k]/env_step[env])))
    plt.yscale('log')
    # plt.grid(color='k', linestyle='--', linewidth=0.5)
    # ax = plt.bar(y_pos+(i*bar_width), inference, bar_width)
    ax = plt.bar(y_pos+(i*bar_width), repeat, bar_width)
    plt.axis([ -0.25, 5,1E-6,1])
    plt.xticks(range(len(objects)), objects)
    # ax.set(ylim=(1E-6, 1E-2))

        

plt.xticks(y_pos, objects)
plt.legend(algs,loc=2)
plt.hlines(env_step[env],-0.25, 5, 'k', '--')
plt.ylabel('Inference Time (seconds)')
plt.xlabel('Number of CPUs')
plt.title('Inference Time in Pusher-v2')

plt.show()









