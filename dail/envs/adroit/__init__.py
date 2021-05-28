from typing import Dict, Union

from dail.envs.adroit.pen_v0 import PenEnvV0
from dail.envs.adroit.door_v0 import DoorEnvV0
from dail.envs.adroit.hammer_v0 import HammerEnvV0
from dail.envs.adroit.relocate_v0 import RelocateEnvV0

# https://github.com/rail-berkeley/d4rl/blob/master/d4rl/infos.py
# Latest commit ec03683 on Feb 9
REF_MIN_SCORE = {
    "pen-human-v0": 96.262799,
    "pen-cloned-v0": 96.262799,
    "pen-expert-v0": 96.262799,
    "door-human-v0": -56.512833,
    "door-cloned-v0": -56.512833,
    "door-expert-v0": -56.512833,
    "hammer-human-v0": -274.856578,
    "hammer-cloned-v0": -274.856578,
    "hammer-expert-v0": -274.856578,
    "relocate-human-v0": -6.425911,
    "relocate-cloned-v0": -6.425911,
    "relocate-expert-v0": -6.425911,
}

REF_MAX_SCORE = {
    "pen-human-v0": 3076.8331017826877,
    "pen-cloned-v0": 3076.8331017826877,
    "pen-expert-v0": 3076.8331017826877,
    "door-human-v0": 2880.5693087298737,
    "door-cloned-v0": 2880.5693087298737,
    "door-expert-v0": 2880.5693087298737,
    "hammer-human-v0": 12794.134825156867,
    "hammer-cloned-v0": 12794.134825156867,
    "hammer-expert-v0": 12794.134825156867,
    "relocate-human-v0": 4233.877797728884,
    "relocate-cloned-v0": 4233.877797728884,
    "relocate-expert-v0": 4233.877797728884,
}

DATASET_URLS = {}

# Adroit v1 envs
for env in ["pen", "door", "hammer", "relocate"]:
    for dataset in ["human", "cloned", "expert"]:
        env_name = f"{env}-{dataset}-v1"

        DATASET_URLS[env_name] = (
            "http://rail.eecs.berkeley.edu/datasets/offline_rl/"
            f"hand_dapg_v1/{env_name}.hdf5"
        )
        REF_MIN_SCORE[env_name] = REF_MIN_SCORE[env + "-human-v0"]
        REF_MAX_SCORE[env_name] = REF_MAX_SCORE[env + "-human-v0"]

infos: Dict[str, Union[Dict[str, str], Dict[str, float]]] = {
    "dataset_urls": DATASET_URLS,
    "ref_min_score": REF_MIN_SCORE,
    "ref_max_score": REF_MAX_SCORE,
}

__all__ = ["DoorEnvV0", "HammerEnvV0", "PenEnvV0", "RelocateEnvV0", "infos"]
