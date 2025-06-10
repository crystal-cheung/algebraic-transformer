import numpy as np
import torch
import pdb
import torch.nn as nn

# Define irreducible representations of Z_n
def zn_irreps(n):
    G = list(range(n))  # group elements 0,1,...,n-1
    reps = []
    for k in range(n):
        # Each rep is a dict from group element g to 1x1 complex matrix exp(2pi i k g / n)
        rep = {g: torch.tensor([[np.exp(2j * np.pi * k * g / n)]], dtype=torch.cfloat) for g in G}
        reps.append(rep)
    return G, reps

# Define transitions for Z_n: identity map modulo n
def zn_transitions(n):
    return {g: g for g in range(n)}


class GroupEmbedding(nn.Module):
    def __init__(self, reps):
        super().__init__()
        self.reps = reps
        self.embedding_dim = sum([rep[0].numel() for rep in reps])

    def forward(self, x):
        batch = []
        for element in x:
            idx = int(element)  # convert tensor to plain Python int
            # blocks = [torch.nan_to_num(rep[idx].flatten().real.to(torch.float32)) for rep in self.reps]
            blocks = [torch.nan_to_num(rep[element].flatten().real.to(torch.float32)) for rep in self.reps]
            batch.append(torch.cat(blocks))
        return torch.stack(batch)

# --- Transition ---
class AlgebraicTransition(nn.Module):
    def __init__(self, reps, transitions):
        super().__init__()
        self.reps = reps
        self.transitions = transitions

    def forward(self, state_embedding, input_symbols):
        batch_size = len(input_symbols)
        new_embeddings = []
        for i in range(batch_size):
            symbol = input_symbols[i]
            t_sigma = self.transitions[symbol]
            # t_sigma = self.transitions[symbol]
            blocks = []
            idx = 0
            for rep in self.reps:
                d = rep[0].shape[0]
                block = state_embedding[i][idx:idx + d*d].reshape(d, d).to(torch.float32)
                block = torch.nan_to_num(block)
                rep_matrix = torch.nan_to_num(rep[t_sigma].real.to(torch.float32))
                rep_matrix = rep_matrix / (rep_matrix.norm() + 1e-6)
                block_updated = block @ rep_matrix

                blocks.append(block_updated.flatten())
                idx += d*d
            new_embeddings.append(torch.cat(blocks))

        new_embeddings_tensor = torch.stack(new_embeddings)

        return new_embeddings_tensor

# --- Layer ---
class AlgebraicTransformerLayer(nn.Module):
    def __init__(self, reps, transitions):
        super().__init__()
        self.transition = AlgebraicTransition(reps, transitions)
        embed_dim = sum([r[0].shape[0]**2 for r in reps])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3, dtype=torch.float32)

    def forward(self, x, input_symbols):
        updated = self.transition(x, input_symbols)
        residual = updated + x
        residual = residual + 1e-6 * torch.randn_like(residual)
        out = self.norm(residual)
        # out = self.norm(updated + x)

        return out

# --- Model ---
class ScalableAlgebraicTransformer(nn.Module):
    def __init__(self, reps, transitions, depth=3):
        super().__init__()
        self.embedding = GroupEmbedding(reps)
        self.layers = nn.ModuleList([
            AlgebraicTransformerLayer(reps, transitions) for _ in range(depth)
        ])
        embed_dim = sum([r[0].shape[0]**2 for r in reps])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, sequences):
        batch_size, seq_len = len(sequences), len(sequences[0])
        # init_state = torch.zeros(batch_size, dtype=torch.long).tolist()
        # x = self.embedding(init_state).to(torch.float32)
        # Use the first digit as input to embedding (for diversity)
        init_state = [seq[0] for seq in sequences]
        x = self.embedding(init_state)

        for t in range(seq_len):
            symbols_t = [seq[t] for seq in sequences]
            for layer in self.layers:
                x = layer(x, symbols_t)


        out = self.output_proj(x).squeeze(-1)

        return out


if __name__ == "__main__":
    # === Define group representations (replace with actual zn_irreps and zn_transitions) ===
    # Assume these functions return reps and transitions for ℤ_10
    G, reps = zn_irreps(10)
    transitions = zn_transitions(10)

    # === Model Setup ===
    model = ScalableAlgebraicTransformer(reps, transitions, depth=3)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = BCEWithLogitsLoss()

    # === Training Loop ===
    num_epochs = 100
    batch_size = 128

    for epoch in range(num_epochs):
        X, y = generate_data_balanced(batch_size)
        input_sequences = X.tolist()

        # Forward pass
        logits = model(input_sequences).squeeze(-1)  # shape: [batch_size]
        loss = criterion(logits, y.float())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accuracy
        acc = ((logits > 0) == y).float().mean().item()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} Acc: {acc:.4f}")
