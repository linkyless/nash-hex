from hexzero.network import PolicyValueNetwork
from hexzero.mcts import Node, MCTS
from hexzero.board import HexBoard
import torch
import torch.nn.functional as F
import numpy as np
import random


TEMPERATURE_MOVES = 14

def get_examples_of_a_match(board_size: int, network: PolicyValueNetwork, simulations: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    examples = []
    Engine = HexBoard(board_size)
    total_cells = board_size * board_size
    count = 1
    if np.random.rand() < 0.5:
        for _ in range(np.random.randint(1, 3)):
            Engine.place(*Engine.index_to_cell(int(np.random.choice(Engine.valid_choices()))))
    
    while Engine.winner is None:
        MonteCarlo = MCTS(Engine, simulations, network)
        (pi, best_move) = MonteCarlo.search()

        black_moves = Engine.current_player() == -1
        temp_pi = pi
        if black_moves:
            temp_pi = pi.reshape(Engine.board_size, Engine.board_size).T.reshape(-1)

        examples.append((Engine.transform_to_canonic_form(), temp_pi))

        best_move = best_move if count > TEMPERATURE_MOVES else int(np.random.choice(total_cells, p=pi))

        (row, col) = Engine.index_to_cell(best_move)
        Engine.place(row, col)

        count += 1

    final_examples = []
    mult = 1
    for board, pi in reversed(examples):
        final_examples.append((board, pi, mult))
        mult = -mult

    # Same boards in 180º
    augmented = []
    for board, pi, z in final_examples:
        augmented.append((board, pi, z))
        augmented.append((board[::-1, ::-1].copy(), pi.reshape(board_size, board_size)[::-1, ::-1].reshape(-1).copy(), z))

    return augmented


def get_value_loss(winners: torch.Tensor, values: torch.Tensor):
    loss = F.mse_loss(winners, values)
    return loss


def get_policy_loss(pi: torch.Tensor, pred_logits: torch.Tensor):
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -torch.sum(pi * log_probs, dim=-1).mean()
    return loss


def train_epoch(network: PolicyValueNetwork, optimizer: torch.optim.Optimizer, examples: list[tuple[np.ndarray, np.ndarray, float]], batch_size: int = 64) -> tuple[float, float]:

    random.shuffle(examples)
    policy_loss_sum = 0
    value_loss_sum  = 0
    batches         = 0

    for i in range(0, len(examples), batch_size):
        batch    = examples[i:i+batch_size]
        boards   = [ex[0] for ex in batch]
        pis       = [ex[1] for ex in batch]
        winners  = [ex[2] for ex in batch]

        boards_t = torch.tensor(np.array(boards), dtype=torch.float32).unsqueeze(1)
        pis       = torch.tensor(np.array(pis), dtype=torch.float32)
        winners  = torch.tensor(np.array(winners), dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        policy_logits, values_pred = network(boards_t)

        policy_loss = get_policy_loss(pis, policy_logits)
        value_loss  = get_value_loss(winners, values_pred)
        total_loss  = policy_loss + value_loss

        total_loss.backward()
        optimizer.step()

        batches += 1
        policy_loss_sum += policy_loss.item()
        value_loss_sum  += value_loss.item()

    return (policy_loss_sum / batches, value_loss_sum / batches)


    
