import time
from pathlib import Path
import torch

from hexzero.network import PolicyValueNetwork
from hexzero.selfplay import get_examples_of_a_match, train_epoch
from hexzero.visualize import show_opening


BOARD_SIZE    = 5
ITERATIONS    = 30
MATCHES       = 50
SIMULATIONS   = 200
EPOCHS        = 5
LEARNING_RATE = 0.001

CHECKPOINTS   = Path("checkpoints")
LOG_FILE      = Path("training_log.csv")
POLICY_LOG    = Path("opening_policies.csv")


if __name__ == "__main__":
    CHECKPOINTS.mkdir(exist_ok=True)

    network   = PolicyValueNetwork(BOARD_SIZE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    if not LOG_FILE.exists():
        LOG_FILE.write_text("iteration,epoch,policy_loss,value_loss\n")

    for iteration in range(ITERATIONS):
        start = time.perf_counter()

        all_examples = []
        for match in range(MATCHES):
            all_examples.extend(get_examples_of_a_match(BOARD_SIZE, network, SIMULATIONS))

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
        
        torch.save(network.state_dict(), CHECKPOINTS / f"iter_{iteration}.pt")

        total_time = time.perf_counter() - start
        print(f"[iter {iteration}] done in {total_time:.1f}s\n")