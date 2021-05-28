from typing import Dict

import numpy as np
import tensorflow as tf

from dail.envs import DomainEnv
from dail.params import SavedParameters

from .model import feedforward, scale_state, scale_action

# DDPG graph imports
from .ddpg.ddpg_graph_with_goal import *


def build_compgraph(
    envs: Dict[str, DomainEnv],
    params: SavedParameters,
    is_transfer: bool = False,
):
    """
    Builds computation graph and defines the train_ops.

    Returns:
        ph: placeholders : tf.placeholders
        targets : targets to fetch during sess.run() call : dict
    """
    env_types = []
    for _, domain_env in envs.items():
        env_types.append(domain_env.env_type)
    assert len(set(env_types)) == 1
    env_type = env_types[0]

    # Placeholder dict
    ph = get_ddpg_ph(envs=envs)

    if env_type == "goal":
        graph = ddpg_graph_with_goal(env, ph, params)
        graph_vars, expert_save_vars, learner_save_vars = get_ddpg_with_goal_vars(env=env)
        targets = get_ddpg_with_goal_targets(
            env=env, ph=ph, graph=graph, var_dict=graph_vars, params=params
        )

    else:
        print(f"[compgraph.py] Unrecognized env type: {env_type}")
        exit(1)

    # Return the placeholders and targets for sess.run() calls
    return ph, graph, targets, expert_save_vars, learner_save_vars


def get_ddpg_ph(env):
    """
    Creates placeholders
    Args:
        env : environments for expert and learner : dict
    Returns:
        ph : placeholders : tf.placeholder
    """

    ph = {}
    for d_, env_params in env.items():
        ph[d_] = {}
        ph[d_]["state"] = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env_params["state_dim"]],
            name=d_ + "_state_ph",
        )
        ph[d_]["next_state"] = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env_params["state_dim"]],
            name=d_ + "_next_state_ph",
        )
        ph[d_]["action"] = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env_params["action_dim"]],
            name=d_ + "_action_ph",
        )
        ph[d_]["action_tv"] = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env_params["action_dim"]],
            name=d_ + "_action_tv_ph",
        )
        ph[d_]["reward"] = tf.placeholder(
            dtype=tf.float32, shape=[None], name=d_ + "_reward_ph"
        )
        ph[d_]["disc_reward"] = tf.placeholder(
            dtype=tf.float32, shape=[None], name=d_ + "_disc_reward_ph"
        )
        ph[d_]["is_not_terminal"] = tf.placeholder(
            dtype=tf.float32, shape=[None], name=d_ + "_is_not_terminal_ph"
        )  # indicators (go into target computation)
        ph[d_]["is_training"] = tf.placeholder(
            dtype=tf.bool, shape=(), name=d_ + "_is_training_ph"
        )  # for dropout
        ph[d_]["raw_action"] = tf.placeholder(
            dtype=tf.float32,
            shape=[None, env_params["action_dim"]],
            name=d_ + "_raw_action_ph",
        )
        ph[d_]["train_disc"] = tf.placeholder(
            dtype=tf.float32, shape=(), name=d_ + "_train_disc_ph"
        )  # for stabilizing gan training
        ph[d_]["gen_weight"] = tf.placeholder(
            dtype=tf.float32, shape=(), name=d_ + "_gen_weight_ph"
        )  # for stabilizing gan training

    return ph
