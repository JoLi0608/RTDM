import time
import gym
import mbrl
import numpy as np
import argparse
import pickle
import copy


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
