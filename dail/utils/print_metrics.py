from typing import Dict, List, Union

import numpy as np


def print_metrics(domain: str, readouts: Dict[str, Union[float, List[float]]]) -> None:
    """Printing the losses from a sess.run() call."""
    spacing = 15

    print_str = (
        "\n"
        f"{domain.capitalize()}>\n"
        f"{'reward':>{spacing}}: {readouts['total_reward']:0.4f}\n"
    )

    if "total_fake_reward" in readouts:
        print_str += f"{'fake reward':>{spacing}}: {readouts['total_fake_reward']:0.4f}\n"

    for key, value in readouts.items():
        if "loss" in key:
            value = np.around(np.mean(value, axis=0), decimals=4)
            print_str += f"{key:>{spacing}}: {value}\n"

    print(print_str)
