from typing import Dict

from dail.envs.make_env import DomainEnv, make_env


# Create a dictionary of environments and return
def make_envs(
    expert_domain: str,
    learner_domain: str,
    seed: int,
) -> Dict[str, DomainEnv]:
    envs = {
        "expert": make_env(expert_domain, seed),
        "learner": make_env(learner_domain, seed),
    }

    return envs
