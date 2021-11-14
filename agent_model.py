import gymcpu

import gym
from gym import make     as gym_make
from gym import register as gym_register
from gym import spec     as gym_spec

from gym.envs.registration import load as load_env_class

import argparse
import json

import os
import signal
import shutil

def read_json(jsonpath):
    with open(jsonpath) as jsonf:
        return json.load(jsonf)

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    agentpath_help = "Path to folder where agent checkpoints are stored."
    parser.add_argument(
        'agentpath', metavar='AGENTPATH',
        help = agentpath_help,
        type = str
    )

    chkpt_help = "Checkpoint of the agent whose model will be checked."
    parser.add_argument(
        'chkpt', metavar='CHKPT',
        help = chkpt_help,
        type = int
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    path = args.agentpath + f'/checkpoint_{args.chkpt}/checkpoint-{args.chkpt}'

    agentconfig = read_json(args.agentpath + '/config.json')
    env = agentconfig['env']
    envconfig = agentconfig['envconfig']

    import ray
    from ray.tune.registry import register_env
    
    Env = load_env_class( gym_spec(env).entry_point )
    register_env(env, lambda config: Env(**config))

    ray.init(ignore_reinit_error=True)
    
    import ray.rllib.agents.ppo as ppo

    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = 0
    config['log_level'] = 'WARN'
    config['env_config'] = envconfig
    agent = ppo.PPOTrainer(config, env=env)
    agent.restore(path)

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

if __name__ == '__main__':
    main()
