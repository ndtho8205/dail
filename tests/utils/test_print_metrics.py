from typing import Dict, List, Union

from dail.utils import print_metrics


def test_should_work_with_fake_reward(capsys) -> None:  # type: ignore
    domain = "self domain"
    readouts: Dict[str, Union[float, List[float]]] = {
        "total_reward": 10.123456789,
        "total_fake_reward": 10.123456789,
        "train_loss": [0.123456789, 0.123456789],
        "test_loss": [1.123456789, 1.123456789],
    }

    print_metrics(domain, readouts)

    got = capsys.readouterr().out
    expected = """
Self domain>
         reward: 10.1235
    fake reward: 10.1235
     train_loss: 0.1235
      test_loss: 1.1235

"""

    assert got == expected


def test_should_work_without_fake_reward(capsys) -> None:  # type: ignore
    domain = "self domain"
    readouts: Dict[str, Union[float, List[float]]] = {
        "total_reward": 10.123456789,
        "train_loss": [0.123456789, 0.123456789],
        "test_loss": [1.123456789, 1.123456789],
    }

    print_metrics(domain, readouts)

    got = capsys.readouterr().out
    expected = """
Self domain>
         reward: 10.1235
     train_loss: 0.1235
      test_loss: 1.1235

"""

    assert got == expected
