import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import os

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

# Data generation function (you'll need to implement this based on your task)
def generate_data_balanced(batch_size, seq_len=5, modular=10):
    """
    Generate balanced training data for modular arithmetic task.
    This is a placeholder - replace with your actual data generation logic.
    """
    X = torch.randint(0, modular, (batch_size, seq_len))
    # Example: binary classification based on sum being even/odd
    y = (X.sum(dim=1) % 2).long()
    return X, y

class GroupEmbedding(nn.Module):
    def __init__(self, reps):
        super().__init__()
        self.reps = reps
        self.embedding_dim = sum([rep[0].numel() for rep in reps])

    def forward(self, x):
        batch = []
        for element in x:
            idx = int(element)  # convert tensor to plain Python int
            blocks = [torch.nan_to_num(rep[idx].flatten().real.to(torch.float32)) for rep in self.reps]
            batch.append(torch.cat(blocks))
        return torch.stack(batch)

class AlgebraicTransition(nn.Module):
    def __init__(self, reps, transitions):
        super().__init__()
        self.reps = reps
        self.transitions = transitions

    def forward(self, state_embedding, input_symbols):
        batch_size = len(input_symbols)
        new_embeddings = []
        for i in range(batch_size):
            symbol = int(input_symbols[i])  # Ensure it's an int
            t_sigma = self.transitions[symbol]
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
        return torch.stack(new_embeddings)

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
        return out

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
        # Use the first digit as input to embedding
        init_state = [seq[0] for seq in sequences]
        x = self.embedding(init_state)

        for t in range(seq_len):
            symbols_t = [seq[t] for seq in sequences]
            for layer in self.layers:
                x = layer(x, symbols_t)

        out = self.output_proj(x).squeeze(-1)
        return out

def train_algebraic_transformer(modular=10, depth=3, num_epochs=100, batch_size=128, 
                              seq_len=5, lr=3e-4, weight_decay=1e-5, save_path="./models"):
    """
    Train an algebraic transformer for modular arithmetic.
    
    Args:
        modular: The modular base (e.g., 10 for Z_10)
        depth: Number of transformer layers
        num_epochs: Number of training epochs
        batch_size: Training batch size
        seq_len: Length of input sequences
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        save_path: Directory to save the trained model
    
    Returns:
        model: Trained model
        training_history: Dictionary with loss and accuracy history
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Setup group representations and transitions
    G, reps = zn_irreps(modular)
    transitions = zn_transitions(modular)
    
    # Initialize model, optimizer, and loss function
    model = ScalableAlgebraicTransformer(reps, transitions, depth=depth)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = BCEWithLogitsLoss()
    
    # Training history
    history = {'loss': [], 'accuracy': []}
    
    print(f"Training Algebraic Transformer for Z_{modular}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Generate training data
        X, y = generate_data_balanced(batch_size, seq_len, modular)
        input_sequences = X.tolist()
        
        # Forward pass
        model.train()
        logits = model(input_sequences).squeeze(-1)
        loss = criterion(logits, y.float())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = (logits > 0).long()
            accuracy = (predictions == y).float().mean().item()
        
        # Store history
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    
    # Save the trained model
    model_save_path = os.path.join(save_path, f"algebraic_transformer_z{modular}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'modular': modular,
        'depth': depth,
        'reps': reps,
        'transitions': transitions,
        'training_history': history,
        'model_config': {
            'modular': modular,
            'depth': depth,
            'seq_len': seq_len,
            'embed_dim': sum([r[0].shape[0]**2 for r in reps])
        }
    }, model_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    return model, history

def load_algebraic_transformer_if_exists(modular, model_dir="./models"):
    """
    Load algebraic transformer for given modular if it exists.
    
    Args:
        modular: The modular base
        model_dir: Directory containing saved models
    
    Returns:
        model: Loaded model or None if not found
        config: Model configuration or None
    """
    model_path = os.path.join(model_dir, f"algebraic_transformer_z{modular}.pth")
    
    if not os.path.exists(model_path):
        print(f"No pre-trained model found for Z_{modular} at {model_path}")
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Reconstruct model
        depth = checkpoint['depth']
        reps = checkpoint['reps']
        transitions = checkpoint['transitions']
        
        model = ScalableAlgebraicTransformer(reps, transitions, depth=depth)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Successfully loaded algebraic transformer for Z_{modular}")
        return model, checkpoint['model_config']
    
    except Exception as e:
        print(f"Error loading model for Z_{modular}: {e}")
        return None, None

if __name__ == "__main__":
    # Train the model
    model, history = train_algebraic_transformer(
        modular=10,
        depth=3,
        num_epochs=100,
        batch_size=128,
        seq_len=5
    )
    
