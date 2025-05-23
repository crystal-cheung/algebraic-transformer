import torch
import torch.nn as nn

class GroupEmbedding(nn.Module):
    def __init__(self, reps):
        super().__init__()
        self.reps = reps  # List of representation matrices [rho^{(1)}, ..., rho^{(k)}]
        self.embedding_dim = sum([r.shape[0] * r.shape[1] for r in reps])

    def forward(self, x):
        # x is a list of group elements, each maps to representation matrices
        # Each element is assumed to be a key to look up representation matrices
        batch = []
        for element in x:
            blocks = [rep[element] for rep in self.reps]  # block matrices
            blocks_flat = torch.cat([b.flatten() for b in blocks])
            batch.append(blocks_flat)
        return torch.stack(batch)

class AlgebraicTransition(nn.Module):
    def __init__(self, reps, transitions):
        super().__init__()
        self.reps = reps  # [rho^{(j)}]
        self.transitions = transitions  # dict of t_sigma for each symbol sigma

    def forward(self, state_embedding, input_symbols):
        batch_size = len(input_symbols)
        new_embeddings = []
        for i in range(batch_size):
            symbol = input_symbols[i]
            t_sigma = self.transitions[symbol]
            blocks = []
            idx = 0
            for rep in self.reps:
                d = rep[0].shape[0]
                block = state_embedding[i][idx:idx + d*d].reshape(d, d)
                block_updated = block @ rep[t_sigma]
                blocks.append(block_updated.flatten())
                idx += d*d
            new_embeddings.append(torch.cat(blocks))
        return torch.stack(new_embeddings)

class AlgebraicTransformerLayer(nn.Module):
    def __init__(self, reps, transitions):
        super().__init__()
        self.transition = AlgebraicTransition(reps, transitions)
        self.norm = nn.LayerNorm(sum([r[0].shape[0]**2 for r in reps]))

    def forward(self, x, input_symbols):
        updated = self.transition(x, input_symbols)
        return self.norm(updated + x)  # residual connection

class ScalableAlgebraicTransformer(nn.Module):
    def __init__(self, reps, transitions, depth=3):
        super().__init__()
        self.embedding = GroupEmbedding(reps)
        self.layers = nn.ModuleList([
            AlgebraicTransformerLayer(reps, transitions) for _ in range(depth)
        ])

    def forward(self, sequence):
        # sequence: list of input symbols
        batch_size = len(sequence)
        init_state = ['id'] * batch_size  # assume initial state is identity element
        x = self.embedding(init_state)
        for layer in self.layers:
            x = layer(x, sequence)
        return x

