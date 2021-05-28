import random
import shutil
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dail.envs import ENVS, make_envs, register_adroit, register_reacher
from dail.utils import create_replay_memory
from dail.agents import ddpg
from dail.params import generate_reacher_params


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Reset the session
    shutil.rmtree(args.logdir, ignore_errors=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    # Register environments
    if "reacher" in [args.expert_domain, args.learner_domain]:
        register_reacher()
    if "adroit" in [args.expert_domain, args.learner_domain]:
        register_adroit()

    # Initialize environments
    envs = make_envs(args.expert_domain, args.learner_domain, args.seed)
    reacher_params = generate_reacher_params(envs=envs)
    replay_memory = create_replay_memory(envs=envs, params=reacher_params)
    print(
        "Environment:\n",
        "\n".join(f"  {domain}:\t{domain_env}" for domain, domain_env in envs.items()),
    )
    print("Using saved parameters:", args.exp_id)

    # Create an agent
    agent = ddpg.DDPGAgent(
        envs=envs,
        params=reacher_params,
        replay_memory=replay_memory,
        logdir=args.logdir,
        # TODO: Check the below args
        cmd_args=args,
        save_expert_dir=args.save_expert_dir,
        save_learner_dir=args.save_learner_dir,
        save_dataset_dir=args.save_dataset_dir,
        load_expert_dir=args.load_expert_dir,
        load_learner_dir=args.load_learner_dir,
        load_dataset_dir=args.load_dataset_dir,
        render=args.render,
        gpu=args.gpu,
        is_transfer=(args.agent_type == "transfer"),
    )

    return

    print("----------------------------")

    if args.agent_type == "gama":
        print("GAMA with")
        print(f"Expert={args.load_expert_dir}")
        print(f"Self={args.save_learner_dir}")
        agent.gama(from_ckpt=False)
    elif args.agent_type == "zeroshot":
        print("Zeroshot Evaluation")
        print(f"Expert={args.load_expert_dir}")
        print(f"Self={args.load_learner_dir}")
        agent.zeroshot()
    elif args.agent_type == "bc":
        print("Behavioral Cloning on Target Expert")
        print(f"Dataset={args.load_dataset_dir}")
        print(f"Save={args.save_expert_dir}")
        agent.bc(num_demo=args.n_demo)
    elif args.agent_type == "expert":
        print("Training Expert")
        print(f"Save={args.save_expert_dir}")
        agent.train_expert(from_ckpt=False)

    elif args.agent_type == "expert_from_ckpt":
        print("Training Expert from Checkpoint")
        print(f"Load={args.load_expert_dir}")
        print(f"Save={args.save_expert_dir}")
        agent.train_expert(from_ckpt=True)

    elif args.agent_type == "create_alignment_taskset":
        print("Creating Alignment Taskset with")
        print(f"Expert={args.load_expert_dir}")
        print(f"Self={args.load_learner_dir}")
        agent.create_alignment_taskset()

    elif args.agent_type == "rollout_expert":
        print(f"Rollout expert({args.load_expert_dir})")
        agent.rollout_expert()
    elif args.agent_type == "create_demo":
        print(f"Create demonstrations dataset and save ({args.save_dataset_dir})")
        agent.create_demonstrations(num_demo=args.n_demo)
    elif args.agent_type == "bc_from_ckpt":
        print("Behavioral Cloning on Target Expert from Checkpoint")
        print(f"Dataset={args.load_dataset_dir}")
        print(f"Load={args.load_expert_dir}")
        print(f"Save={args.save_expert_dir}")
        agent.bc(from_ckpt=True)
    else:
        print("Unrecognized experiment type")
        exit(1)
    print("----------------------------")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dail",
        description="Domain Adaptive Imitation Learning",
    )

    # Domains
    parser.add_argument(
        "--expert_domain",
        choices=ENVS.keys(),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--learner_domain",
        choices=ENVS.keys(),
        required=True,
        type=str,
    )
    parser.add_argument("--seed", default=0, type=int)

    # Agent
    parser.add_argument(
        "--agent_type",
        choices=["gama", "zeroshot", "bc"],
        required=True,
        type=str,
    )
    parser.add_argument(
        "--algo",
        choices=["ddpg"],
        required=True,
        type=str,
    )

    # Log
    current_dir = Path.cwd()
    default_log_dir = current_dir / "logs" / "temp"
    parser.add_argument("--logdir", default=default_log_dir, type=Path)

    # TODO: Check the below args
    parser.add_argument("--load_expert_dir", default="./saved_expert/temp", type=str)
    parser.add_argument("--load_learner_dir", default="./saved_learner/temp", type=str)
    parser.add_argument("--load_dataset_dir", default="./temp/empty.pickle", type=str)

    parser.add_argument("--save_expert_dir", default="./saved_expert/temp", type=str)
    parser.add_argument("--save_learner_dir", default="./saved_learner/temp", type=str)
    parser.add_argument("--save_dataset_dir", default="./temp/temp.pickle", type=str)

    parser.add_argument("--expert_dataset_dir", default="./saved_dataset/temp", type=str)
    parser.add_argument("--learner_dataset_dir", default="./saved_dataset/temp", type=str)

    parser.add_argument("--exp_id", default="reacher", type=str)
    parser.add_argument("--doc", default="", type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--render", default=1, type=int)
    parser.add_argument("--n_demo", type=int, default=0)

    return parser
