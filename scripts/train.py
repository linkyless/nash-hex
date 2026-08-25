import os
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from hexzero.network import PolicyValueNetwork
from hexzero.selfplay import get_examples_of_a_match, train_epoch
from hexzero.visualize import show_opening
from hexzero.probe import build_probe_positions, policy_diagnostics


BOARD_SIZE    = 5
ITERATIONS    = 100
MATCHES       = 50
SIMULATIONS   = 600
EPOCHS        = 2
LEARNING_RATE = 0.001
BUFFER_ITERATIONS = 10
WORKERS       = os.cpu_count()

CHECKPOINTS   = Path("checkpoints")
LOG_FILE      = Path("training_log.csv")
POLICY_LOG    = Path("opening_policies.csv")
DIAG_FILE     = Path("policy_diagnostics.csv")


def _play_one(state_dict):
    torch.set_num_threads(1)
    np.random.seed(os.getpid() ^ int.from_bytes(os.urandom(2), "little"))
    network = PolicyValueNetwork(BOARD_SIZE)
    network.load_state_dict(state_dict)
    network.eval()
    return get_examples_of_a_match(BOARD_SIZE, network, SIMULATIONS)


if __name__ == "__main__":
    CHECKPOINTS.mkdir(exist_ok=True)

    network   = PolicyValueNetwork(BOARD_SIZE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    buffer    = deque(maxlen=BUFFER_ITERATIONS)
    probe     = build_probe_positions(BOARD_SIZE)

    if not LOG_FILE.exists():
        LOG_FILE.write_text("iteration,epoch,policy_loss,value_loss\n")
    if not DIAG_FILE.exists():
        DIAG_FILE.write_text("iteration,effective_moves,value_confidence\n")

    print(f"usando {WORKERS} procesos para el self-play")

    for iteration in range(ITERATIONS):
        start = time.perf_counter()

        state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
        fresh = []
        with Pool(WORKERS) as pool:
            for chunk in pool.imap_unordered(_play_one, [state_dict] * MATCHES):
                fresh.extend(chunk)
        buffer.append(fresh)

        all_examples = [ex for piece in buffer for ex in piece]

        selfplay_time = time.perf_counter() - start
        print(
            f"[iter {iteration}] {len(all_examples)} ejemplos "
            f"en {selfplay_time:.1f}s"
        )

        for epoch in range(EPOCHS):
            p_loss, v_loss = train_epoch(network, optimizer, all_examples)
            print(
                f"[iter {iteration}] epoch {epoch}: "
                f"policy={p_loss:.4f} value={v_loss:.4f}"
            )
            with LOG_FILE.open("a") as f:
                f.write(f"{iteration},{epoch},{p_loss:.6f},{v_loss:.6f}\n")

        print(f"[iter {iteration}] data:")
        pi = show_opening(network, BOARD_SIZE, SIMULATIONS)

        with POLICY_LOG.open("a") as f:
            f.write(f"{iteration}," + ",".join(f"{p:.6f}" for p in pi) + "\n")

        eff_moves, val_conf = policy_diagnostics(network, probe)
        print(f"[iter {iteration}] jugadas efectivas={eff_moves:.2f} |value|={val_conf:.2f}")
        with DIAG_FILE.open("a") as f:
            f.write(f"{iteration},{eff_moves:.4f},{val_conf:.4f}\n")

        torch.save(network.state_dict(), CHECKPOINTS / f"iter_{iteration}.pt")

        total_time = time.perf_counter() - start
        print(f"[iter {iteration}] done in {total_time:.1f}s\n")