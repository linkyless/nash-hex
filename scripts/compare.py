from hexzero.arena import get_win_rate
from hexzero.network import PolicyValueNetwork
import torch

NUMBER_OF_MATCHES = 20
SIMULATIONS       = 400


def compare_networks(board_size: int) -> None:
    net1 = PolicyValueNetwork(5)
    net1.load_state_dict(torch.load("checkpoints/iter_10.pt"))
    net1.eval()

    net2 = PolicyValueNetwork(5)
    net2.load_state_dict(torch.load("checkpoints/iter_0.pt"))
    net2.eval()

    (win_rate_NET1, win_rate_NET2) = get_win_rate(net1, net2, board_size, SIMULATIONS, NUMBER_OF_MATCHES)

    print(f"Win-rate network 1: {win_rate_NET1}")
    print(f"Win-rate network 2: {win_rate_NET2}")

if __name__ == "__main__":
    compare_networks(5)
