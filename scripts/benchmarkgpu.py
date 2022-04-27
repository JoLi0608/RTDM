import time
import gym
import mbrl
import numpy as np
import argparse
import pickle
import copy
import math
# Input arguments from command line.
parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path", required=True, help="Filepath to trained checkpoint",
                    default="/app/data/ray_results/2/ARS_CartPole-v0_661d3_00000_0_2022-03-31_10-07-40/checkpoint_000100/checkpoint-100")
parser.add_argument("--algo", required=True, help="Algorithm used", default="ARS")
parser.add_argument("--cpu", required=True, help="Number of CPU", default=8)
parser.add_argument("--gpu", required=True, help="Number of GPU", default=0)

args = vars(parser.parse_args())


def load(path,algo,env_name="Hopper-v2",gpu=False):
    if gpu:
        print("Using GPU")
    else:
        print("Using CPU")
    import numpy as np
    if algo == "mbpo":
        if env_name == "Pusher-v2":
            import mbrl.env.pets_pusher as pusher
            env = pusher.PusherEnv()
        elif env_name == "Humanoid-v2":
            import mbrl.env.humanoid_truncated_obs as humanoid
            env = humanoid.HumanoidTruncatedObsEnv()
        else:
            env = gym.make(env_name)

        from mbrl.planning.core import load_agent
        device = "cuda" if gpu else "cpu"
        ag = load_agent(path, env,device)
        agent = lambda obs: ag.act(obs)
    elif algo == "planet":
        import omegaconf
        import torch
        import hydra
        from mbrl.models import ModelEnv, ModelTrainer
        from mbrl.env.termination_fns import no_termination
        from mbrl.planning import RandomAgent, create_trajectory_optim_agent_for_model
        from mbrl.util.env import EnvHandler
        import numpy as np
        import os
        device = "cuda" if gpu else "cpu"
        os.environ["MUJOCO_GL"] = "osmesa"
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        cfg["device"] = "cpu"
        env, term_fn, reward_fn = EnvHandler.make_env(cfg)
        torch_generator = torch.Generator(device=cfg.device)
        cfg.dynamics_model.action_size = env.action_space.shape[0]
        planet = hydra.utils.instantiate(cfg.dynamics_model)
        planet.load(path)
        planet.to(device)
        model_env = ModelEnv(env, planet, no_termination, generator=torch_generator)
        ag = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)

        def agent(obs,done=False):
            import numpy as np
            if done:
                print("reset agent")
                ag.reset()
                planet.reset_posterior()
                planet.update_posterior(obs, action=None, rng=torch_generator)
            return np.clip(ag.act(obs),-1,1)

    elif algo == "pets":
        import mbrl.util.env
        import mbrl.util.common
        import mbrl.planning
        import omegaconf
        import torch
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        if not gpu:
            cfg["device"] = "cpu"
        torch_generator = torch.Generator(device=cfg.device)
        env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
        dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, env.observation_space.shape, env.action_space.shape)
        dynamics_model.load(path)
        if gpu == False:
            dynamics_model.to("cpu")
        model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
        ag = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
        action = ag.act(env.reset())
        action = np.clip(action, -1.0, 1.0)  # to account for the noise
        agent = lambda obs: np.clip(ag.act(obs),-1.0,1.0)
    elif algo == "rtrl":
        import rtrl
        r = rtrl.load(path+"state")
        print("GPU : ",gpu)

        if int(gpu):
            r.agent.model.to("cuda")
        else:
            r.agent.model.to("cpu")

        agent = lambda obs: r.agent.act(obs,[],[],[],train=False)[0]
        env = gym.make(env_name)
    elif algo == "ars":
        env = gym.make(env_name)
        tmp = np.random.uniform(size=(env.observation_space.shape[0],env.action_space.shape[0]))
        agent = lambda obs: np.dot(obs,tmp)
    elif algo == "sac":
        from spinup.utils.test_policy import load_policy_and_env, run_policy
        device = "cuda" if gpu else "cpu"
        env,agent = load_policy_and_env(path,device=device)
    elif algo == "ppo":
        from spinup.utils.test_policy import load_policy_and_env, run_policy
        device = "cuda" if gpu else "cpu"
        env,agent = load_policy_and_env(path,device=device)
    else:
        print("Algo not known", algo)
    return agent,env

  
def create_episode(agent,env,num_steps=100,conc_prev=False,cpu=1):
    done = True
    t1 = time.time()
    rep = 0
    obs_list = []
    tmp_compute = np.array([0])
    while rep < 10:
        if done:
            obs = env.reset()
            prev_action = np.zeros(env.action_space.shape[0])    
        if conc_prev:
            obs = (obs,prev_action)
        obs_list.append(copy.deepcopy(obs))
        tmp_t1 = time.time()
        if args["algo"] == "planet":
            action = agent(obs,done)  # Get action
        else:
            action = agent(obs)  # Get action
        tmp_compute = np.append(tmp_compute,[time.time()-tmp_t1])
        obs,_ ,done , _ = env.step(action)
        prev_action = action
        rep += 1    
    pickle.dump(obs_list,open(args["path"]+"dump_episode.pkl","wb"))
    
    
def run_env(agent,env,num_steps=100,conc_prev=False,cpu=1):
    
    obs_list = pickle.load(open(args["path"]+"dump_episode.pkl","rb"))
    if args["algo"] == "planet":
        _ = agent(env.reset(),True)
        
    t1 = time.time()
    acc = 0
    while time.time()-t1 < 1:
        _ = [agent(obs) for obs in obs_list]
        acc += 1
    
    print(acc)
    if acc > 0:
      print("Expand obs by factor of ",acc," from ",len(obs_list))
      obs_list = obs_list * acc

      
    compute_time = []
    for i in range(10):
        t1 = time.time()
        result = [agent(obs) for obs in obs_list]
        compute_time.append((time.time()-t1)/float(len(obs_list)))
        print(time.time()-t1)
    return np.array(compute_time)


for i in ["HalfCheetah-v2","Hopper-v2","continuous_CartPole-v0","Humanoid-v2","Pusher-v2","dmc_walker_walk","dmc_cartpole_balance","dmc_cheetah_run"]:
    if i in args["path"]:
        env_name = i


agent,env = load(args["path"],args["algo"],env_name=env_name,gpu=int(args["gpu"]))

store_episode = False
conc_prev = True if args["algo"] == "rtrl" else False


if store_episode:
    create_episode(agent,env,conc_prev=conc_prev,cpu=float(args["cpu"]))
else:
    t = run_env(agent,env,conc_prev=conc_prev,cpu=float(args["cpu"]))
    path_inference = "/app/data/inference_time/data.pkl"
    try:
        f = open(path_inference,"rb")
        inf_time = pickle.load(f)
        f.close()
    except:
        inf_time = {}

    name_experiment = "cpu_"+args["cpu"]+"_"+args["algo"]+"_"+env_name+"_gpu_"+args["gpu"]
    inf_time[name_experiment] = t
    print("Algo: ",args["algo"], " env: ",env_name)
    print("Mean time: ", t.mean()," median: ",np.median(t))

    f = open(path_inference,"wb")
    pickle.dump(inf_time,f)
    f.close()


#elif algo == "ars":
#     path = "/app/data/ray_results/1/ARS_Hopper-v2_47175_00000_0_2022-04-07_11-24-24/checkpoint_015300/checkpoint-15300"
#     num_gpus = 0.2 if gpu else 0
#     trainer = ars.ARSTrainer(config={
#             "framework": "torch",
#             "num_gpus": num_gpus,},env="Hopper-v2")
#     trainer.restore(path)
#     agent = lambda obs: trainer.compute_single_action(obs)
#     env = gym.make("Hopper-v2")
