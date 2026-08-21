from hexzero.arena import get_win_rate
from hexzero.network import PolicyValueNetwork
import torch
import numpy as np
import math

NUMBER_OF_MATCHES = 100
SIMULATIONS       = 200
NUMBER_OF_NETS    = 30
STEP              = 5

def win_rate_to_delta_elo(win_rate: float) -> float:
    p = np.clip(win_rate, 0.01, 0.99)
    return -400.0 * math.log10((1.0 / p) - 1.0)

def compare_networks(board_size: int) -> None:
    net1 = PolicyValueNetwork(5)
    net1.load_state_dict(torch.load("checkpoints/iter_29.pt"))
    net1.eval()

    net2 = PolicyValueNetwork(5)
    net2.load_state_dict(torch.load("checkpoints/iter_0.pt"))
    net2.eval()

    (win_rate_NET1, win_rate_NET2) = get_win_rate(net1, net2, board_size, SIMULATIONS, NUMBER_OF_MATCHES)

    print(f"Win-rate network 1: {win_rate_NET1}")
    print(f"Win-rate network 2: {win_rate_NET2}")


def free_for_all(board_size: int) -> None:
    elo_ratings = {0: 0.0}

    curr_net = PolicyValueNetwork(board_size)
    curr_net.eval()
    next_net = PolicyValueNetwork(board_size)
    next_net.eval()

    for i in range(0, NUMBER_OF_NETS - STEP, STEP):
        base_idx = i
        next_idx = i + STEP

        base_path = f"checkpoints/iter_{base_idx}.pt"
        next_path = f"checkpoints/iter_{next_idx}.pt"

        curr_net.load_state_dict(torch.load(base_path, map_location="cpu"))
        next_net.load_state_dict(torch.load(next_path, map_location="cpu"))

        win_rate_curr, win_rate_next = get_win_rate(curr_net, next_net, board_size, SIMULATIONS, NUMBER_OF_MATCHES)
        delta_elo = win_rate_to_delta_elo(win_rate_next)
        curr_elo  = elo_ratings[base_idx] + delta_elo
        elo_ratings[next_idx] = curr_elo

    print(elo_ratings)

if __name__ == "__main__":
    free_for_all(5)
