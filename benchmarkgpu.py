import time
import gym
import mbrl
import numpy as np
import argparse
import pickle


# Input arguments from command line.
parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument("--path", required=True, help="Filepath to trained checkpoint",
                    default="/app/data/ray_results/2/ARS_CartPole-v0_661d3_00000_0_2022-03-31_10-07-40/checkpoint_000100/checkpoint-100")
parser.add_argument("--algo", required=True, help="Algorithm used", default="ARS")

args = vars(parser.parse_args())


def load(path,algo,env_name="Hopper-v2",gpu=False):
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
        if not gpu:
            r.agent.model.to("cpu")
        else:
            r.agent.model.to("cuda")
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
    
def run_env(agent,env,num_steps=1000,conc_prev=False):
    done = True
    compute_time = []
    for i in range(num_steps):
        if done:
            obs = env.reset()
            prev_action = np.zeros(env.action_space.shape[0])
        if conc_prev:
            obs = (obs,prev_action)
        t1 = time.time()
        action = agent(obs)  # Get action
        compute_time.append(time.time()-t1)
        obs,_ ,done , _ = env.step(action)
        prev_action = action
    return np.array(compute_time)


for i in ["HalfCheetah-v2","Hopper-v2","continuous_CartPole-v0","Humanoid-v2","Pusher-v2"]:
    if i in args["path"]:
        env_name = i


agent,env = load(args["path"],args["algo"],env_name=env_name)

if args["algo"] == "rtrl":
    t = run_env(agent,env,conc_prev=True)
else:   
    t = run_env(agent,env)

try:
    f = open("/app/RTDM/scripts/inf_time.pkl","rb")
    inf_time = pickle.load(f)
    f.close()
except:
    inf_time = {}

inf_time[args["algo"]+"_"+env_name] = t
print("Algo: ",args["algo"], " env: ",env_name)
print("Mean time: ", t.mean()," median: ",t.median())

f = open("/app/RTDM/scripts/inf_time.pkl","wb")
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
