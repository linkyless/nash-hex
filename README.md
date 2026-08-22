[![Try the demo →](https://img.shields.io/badge/Try_the_demo_→-1a1a1a?style=for-the-badge&logo=huggingface&logoColor=white)](#)
[![Notes](https://img.shields.io/badge/Notes-obsidian-1a1a1a?style=for-the-badge&logo=obsidian&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.13-1a1a1a?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-1a1a1a?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

# nash

Nash plays Hex on a `5x5` board and it beats its own first iteration `97` games out of `100`. Nobody taught it anything beyond the rules, so there was no opening book for it to memorise and no evaluation function that I wrote by hand telling it which positions are good. It started from random weights, played `1500` games against itself, and everything it knows came out of that. You can play against it [here](#).

The algorithm is AlphaZero, scaled down until it runs on a laptop. A **Monte Carlo tree search** chooses the move by walking down the game tree and keeping count of what it finds, and a **convolutional network** with two outputs guesses who is ahead and where to play, which the search consults every time it reaches a position it has not seen before. The reason any of this bootstraps from nothing is that the search always plays better than the network steering it, so the search can be used as a teacher, and a network that has learned from a better teacher then makes the next search better still.

![the board](assets/board.gif)

## state representation and network topology

The board is a plain `5x5` matrix, and I fill it with a `1` wherever the player about to move has a stone, a `-1` where the opponent has one, and a `0` on the empty cells. This is the **canonical form**, and the point of it is that the network only ever has to answer one question, which is whether the player to move is winning.

Hex makes that awkward, because one player wins by connecting north to south and the other by connecting east to west, so an identical arrangement of stones means something different depending on whose turn it is. What I do about it is transpose the matrix on the second player's turn, which converts their goal into the first player's goal. Formally, the map applied to a board $B$ when it is the second player's turn is

$$\phi(B) = (-B)^\top$$

and that only works if the geometry survives it, so I checked before relying on it. A cell has six neighbours, at offsets

$$D = \{(0,1),\,(0,-1),\,(1,0),\,(-1,0),\,(-1,1),\,(1,-1)\}$$

Transposition sends $(r,c) \mapsto (c,r)$. Applying it to $D$ shuffles the four orthogonal offsets among themselves and makes the two diagonals trade places, so $\phi(D) = D$ and the adjacency graph is untouched. The transposition turns out to be *the* symmetry that swaps the roles of the two players, which is why composing it with the sign flip gives a canonical form that is exact rather than approximate.

The network is small. Three convolutional layers with `16` `3x3` filters each and a **ReLU** between them, because stacking convolutions with nothing in between leaves you with a composition of linear maps, which is still one linear map. The receptive field after $L$ layers of kernel size $k$ is

$$R = 1 + L(k-1)$$

so three layers of `3x3` reach $R = 7$, wide enough to cover a `5x5` board from any cell on it. On `11x11` it would not be, and that matters more in Hex than it would in image classification, since whether a connection between two edges is viable can depend on a stone at the far end of the board.

The trunk then splits into two heads. The **value head** compresses the board into one number through `tanh`, where $-1$ means the player to move is certainly losing. The **policy head** produces one number per cell, with the occupied cells set to $-\infty$ before the softmax so they come out at exactly zero afterwards, since $e^{-\infty} = 0$ leaves the remaining mass to renormalise itself over the legal moves.

## tree search and the exploration term

A plain Monte Carlo tree search evaluates a position by playing random games from it until somebody wins. It does work, and it is extremely noisy, because a random continuation from a position that is objectively winning still loses about half the time.

The rule that decides which branch to walk down is **UCB1**, and it did not come from game playing at all. It comes from the multi armed bandit problem, the one with several slot machines of unknown payout and a limited number of coins.

$$\frac{W}{N} + c\sqrt{\frac{\ln N_{\text{parent}}}{N}}$$

The idea is that you stop comparing options by the average payout you have observed and start comparing them by the highest value they could plausibly have. A child visited `3` times has a wide margin of error, so its optimistic ceiling stays high even when the average is mediocre, whereas a child visited `300` times has a narrow one. And then the useful consequence, which is that a node can win the comparison either because it genuinely is good or because you have barely looked at it and the uncertainty is doing the work. Exploitation and exploration both fall out of one rule. The square root is the width of that confidence interval, decaying as $O(1/\sqrt{N})$, and the logarithm is what keeps the bonus growing slowly enough that the search eventually commits to something.

With a network attached the random games disappear. When the search reaches an unexpanded position it queries the network once and gets both answers at the same time. The value replaces the rollout. The policy becomes a prior on every child, so the first visits head towards the moves the network considers worth examining instead of spreading evenly over `25` cells that look identical to a search with no information. The rule that combines them is **PUCT**, where the $P$ is that prior.

$$Q(s,a) + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

Reading the shape rather than the symbols helps here. At $N(s,a) = 0$ the term $Q$ is empty and the prior is the only information in existence, so the prior decides alone. As visits accumulate the denominator grows and the prior's contribution decays as $O(1/N(s,a))$ while the empirical average takes over, so the intuition dissolves at the rate the search finds things out on its own. A network that was confidently wrong can therefore be overruled.

## supervision without a dataset

Training the value head is easy enough. You play a game to the end, and every position where the eventual winner was the one to move gets labelled $z = +1$ while the others get $z = -1$. That result is the only thing in the entire system that does not come out of the network.

The policy head is the interesting one, because training it the usual supervised way would require knowing the correct move, and nobody knows the correct move in Hex.

Well, the search has already produced one without being asked to. After a few hundred simulations the visits are not spread evenly across the children, they have piled up on whichever moves the search found worth pursuing, so normalising the counts

$$\pi(a) = \frac{N(s,a)}{\sum_b N(s,b)}$$

gives a distribution over the same `25` cells the policy head outputs. And $\pi$ is better than the network's own $p$, because it has a few hundred simulations of real lookahead baked into it. So the network gets trained to predict, instantly, the conclusion it would have arrived at if it had stopped and thought about it.

$$L = (z - v)^2 - \pi^\top \log p$$

Squared error on the left, cross entropy on the right. There is one thing about that second term that took me a while to understand, which is that a cross entropy measured against some distribution can never drop below the entropy of that distribution,

$$H(\pi, p) \ge H(\pi)$$

with equality only when $p = \pi$. A flat policy over `25` cells has $H = \ln 25 \approx 3.22$, so during the early iterations the loss is *physically incapable* of going below that number however well the network learns. Seeing it drop under `3.22` later on tells you something else, which is that the labels themselves have become sharper.

![training losses](results/loss.png)

## two sign errors

I had two sign errors and neither of them crashed anything. What exposed them was putting a trained network up against a freshly initialised one and counting. `40%`.

Error #1: On the second player's turn the network is looking at a transposed board, so the policy it hands back is indexed in transposed coordinates, and I was reading it with indices taken from the real board. For half of every game the priors were scrambled, and a prior that is wrong in a consistent way does more damage than a random one, since the search will average noise out over enough visits and has no way to average out a systematic error.

Error #2: A node stores its value from the point of view of whoever made the move that led to it, and the network hands back its value from the point of view of whoever moves next, and those are opposite people. Terminal nodes were unaffected, since they return a hardcoded $+1$, so the search kept finding mate in one perfectly well, and that is precisely why the first test I wrote reported the sign as correct and sent me looking somewhere else for a day.

The better the value head became at judging positions, the more consistently the search picked the wrong side of them. The untrained network was winning because its value output was noise, and noise at least has no direction to it. Haha, it was nice but frustrating.

![before and after](results/milestones.png)

## measurements

`30` iterations, `50` self-play games each, `200` simulations per move, `5` training passes per batch, Adam at `0.001`. A bit under `2h`, running entirely on CPU. Moving a network this small to the GPU buys nothing until the evaluations are batched, since the bottleneck is the tree search in interpreted Python.

The final checkpoint beats the first one `97` games out of `100`, over `50` paired openings where each starting position is played twice with the colours swapped. Converting a win rate to a rating difference through

$$\Delta_{\text{elo}} = -400 \log_{10}\left(\frac{1}{p} - 1\right)$$

puts that at roughly `600` Elo.

![elo](results/elo.png)

The chained ladder in that figure adds up to more than `1100`, and the direct measurement is the one I trust. Elo is not transitive between engines, so measuring adjacent pairs and adding the differences accumulates error in a single direction. Each rung is `30` games, so any single step should be read as a tendency.

That shape dips between iteration `10` and `15`, and the loss curves point at why. The value loss sits at almost exactly zero throughout that stretch, which sounds like the network having solved its problem and means the opposite. The self-play games had collapsed into near identical repetitions of one another, so predicting the winner had become trivial and there was nothing left in the data worth learning. Around iteration `21` the temperature sampling opens a different line and the ladder starts climbing again.

![opening policies](results/openings.png)

That is the visit distribution over an empty board, one panel per iteration, all normalised against the same peak. Iteration `0` spreads `200` simulations across `25` cells and learns almost nothing from any of them. By the end nearly all of them land on a single square. That concentration is what makes the search strong and it is also what starves the training data of variety.

## structure

    nash-hex/
    ├── hexzero/
    │   ├── board.py       # HexBoard, win detection, canonical form
    │   ├── ufds.py        # union-find with path compression
    │   ├── mcts.py        # Node class and the search
    │   ├── network.py     # policy head, value head
    │   ├── selfplay.py    # game generation and the training epoch
    │   ├── arena.py       # head to head matches with paired openings
    │   ├── visualize.py   # terminal heatmap of the opening policy
    │   └── cli.py         # a random agent playing with the curr network
    ├── scripts/
    │   ├── train.py       # the self-play loop which writes checkpoints and logs
    │   ├── ladder.py      # plays the checkpoints against each other
    │   └── plots.py       # reads the logs and draws stuff
    ├── web/
    │   ├── app.py         # FastAPI, stateless
    │   └── static/        # pixel board on a canvas
    ├── tests/
    └── results/

Notes:
- Win detection runs on union-find with `4` virtual nodes, one per edge. Every stone joins its neighbours of the same colour and any edge node it touches, so asking whether white has won amounts to asking whether the north and south nodes ended up in the same set. With path compression and union by size that is $O(\alpha(n))$ per query instead of a search over the board, which matters once the tree calls it hundreds of thousands of hundreds times.
- The first version that worked took `26s` to generate `5` games, and the profiler put `17` of those inside `copy.deepcopy`. Writing a `copy` method that knows there is exactly one array and two union-find structures to duplicate brought it down to `9.5s`. My guess about where the time was going, before I ran the profiler, was wrong.
- The web version holds no session state.

## running it

    pip install -r requirements.txt
    py -3.13 -m scripts.train              # a couple of hours, writes checkpoints/
    py -3.13 -m scripts.ladder             # plays the checkpoints against each other
    py -3.13 -m scripts.plots              # the plots
    py -3.13 -m uvicorn web.app:app        # you can play against it at localhost:8000

## what is missing
Obviously, there's always something missing from projects like this, and above all, they form part of a conceptual framework which – as I’m bound to overlook something – I failed to pick up on.
The following were the 'missing elements' after I'd analysed the overall picture dispassionately.

- **Dirichlet noise** on the root priors during self-play, which is the standard mechanism against the policy collapsing onto one line and the thing that would most directly address the dip at iteration `12`.
- **Gating**, so a new checkpoint replaces the current one only after clearing some threshold against it.
- A **replay buffer** holding several iterations of games instead of only the most recent one. More than `8` moves of temperature sampling. 
- After that, an **exact solver**. Hex on `5x5` is small enough to solve with alpha beta and memoisation over the same union-find, which would let the engine be measured against perfect play instead of against its own first iteration. Then batched network evaluation, and residual blocks, which three layers on `5x5` have no need for and `11x11` would.

## where this came from

For the tree search, the survey by Browne et al., *A Survey of Monte Carlo Tree Search Methods* (2012), and for UCB1 and the bandit framing underneath it, Auer, Cesa-Bianchi and Fischer. For the self-play loop and PUCT, Silver et al., *Mastering the game of Go without human knowledge* (2017). Piet Hein invented Hex in 1942, and John Nash later proved that the first player has a winning strategy using a strategy stealing argument. The engine is named after him.

A longer writeup in Spanish is on the way, structured as an Obsidian vault like the one I put together for [the neural network](https://linkyless.github.io/neural-network-notes). It will go [here](#) once it exists.