# train.py
#
# PURPOSE: This is the Python training script that each worker executes.
#
# HOW IT FITS IN THE SYSTEM:
# ──────────────────────────
#   1. Go worker receives a task from master (shard path, batch size, learning rate)
#   2. Go worker launches this script as a subprocess:
#        python train.py --data=dataset/shard1.txt --batch_size=2 --lr=0.01 --weights=weights.json
#   3. This script:
#        a) Loads the data shard
#        b) Loads current model weights
#        c) Runs one training step (forward pass → loss → backward pass)
#        d) Outputs gradients as JSON to stdout
#   4. Go worker reads the JSON gradients and sends them to master via gRPC
#
# THE MODEL:
# ──────────
# We use a simple 2-layer neural network (not a full transformer yet).
# This keeps training fast on CPU while still demonstrating real distributed training.
#
#   Input (text) → Tokenize → Embedding → Linear Layer → Output → Loss
#
# The gradients are REAL mathematical gradients computed via backpropagation.
# The master averages them across workers, which is EXACTLY how distributed
# training works in PyTorch DDP, Horovod, etc.

import argparse
import json
import sys
import math
import os

# ──────────────────────────────────────────────────────────
# TOKENIZER — converts text to numbers
# ──────────────────────────────────────────────────────────
# Real LLMs use BPE (Byte-Pair Encoding) tokenizers.
# We use a simple character-level tokenizer for simplicity.
# Each unique character gets a number (0-255).

def tokenize(text, vocab_size=50):
    """Convert text to list of integer token IDs."""
    tokens = []
    for ch in text:
        tokens.append(ord(ch) % vocab_size)
    return tokens


# ──────────────────────────────────────────────────────────
# MODEL — simple neural network with trainable weights
# ──────────────────────────────────────────────────────────
# Architecture:
#   Input tokens → Embedding lookup → Hidden layer → Output → Loss
#
# Weight shapes:
#   embedding:  [vocab_size x embed_dim]     — converts token IDs to vectors
#   hidden_w:   [embed_dim x hidden_dim]     — first linear layer
#   hidden_b:   [hidden_dim]                 — bias
#   output_w:   [hidden_dim x vocab_size]    — predicts next token
#   output_b:   [vocab_size]                 — bias

class SimpleModel:
    def __init__(self, vocab_size=50, embed_dim=16, hidden_dim=32):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Initialize weights (small random-like values using a simple seed)
        self.weights = self._init_weights()

    def _init_weights(self):
        """Create initial weight vector. All model parameters flattened into one list."""
        # Total parameters:
        #   embedding:  vocab_size * embed_dim     = 50 * 16 = 800
        #   hidden_w:   embed_dim * hidden_dim     = 16 * 32 = 512
        #   hidden_b:   hidden_dim                 = 32
        #   output_w:   hidden_dim * vocab_size    = 32 * 50 = 1600
        #   output_b:   vocab_size                 = 50
        #   TOTAL = 2994 parameters
        total = (self.vocab_size * self.embed_dim +
                 self.embed_dim * self.hidden_dim +
                 self.hidden_dim +
                 self.hidden_dim * self.vocab_size +
                 self.vocab_size)

        # Initialize with small values (Xavier-like initialization)
        weights = []
        for i in range(total):
            # Deterministic pseudo-random initialization
            val = math.sin(i * 0.01) * 0.1
            weights.append(val)
        return weights

    def load_weights(self, weights):
        """Load weights from a flat list (received from master)."""
        self.weights = weights[:]

    def get_weight_count(self):
        """Return total number of trainable parameters."""
        return len(self.weights)

    def _get_embedding(self):
        """Extract embedding matrix from flat weight vector."""
        size = self.vocab_size * self.embed_dim
        flat = self.weights[:size]
        # Reshape to [vocab_size][embed_dim]
        matrix = []
        for i in range(self.vocab_size):
            row = flat[i * self.embed_dim:(i + 1) * self.embed_dim]
            matrix.append(row)
        return matrix

    def _get_hidden(self):
        """Extract hidden layer weights and bias."""
        offset = self.vocab_size * self.embed_dim
        w_size = self.embed_dim * self.hidden_dim
        flat_w = self.weights[offset:offset + w_size]
        offset += w_size
        bias = self.weights[offset:offset + self.hidden_dim]

        # Reshape W to [embed_dim][hidden_dim]
        matrix = []
        for i in range(self.embed_dim):
            row = flat_w[i * self.hidden_dim:(i + 1) * self.hidden_dim]
            matrix.append(row)
        return matrix, bias

    def _get_output(self):
        """Extract output layer weights and bias."""
        offset = (self.vocab_size * self.embed_dim +
                  self.embed_dim * self.hidden_dim +
                  self.hidden_dim)
        w_size = self.hidden_dim * self.vocab_size
        flat_w = self.weights[offset:offset + w_size]
        offset += w_size
        bias = self.weights[offset:offset + self.vocab_size]

        # Reshape to [hidden_dim][vocab_size]
        matrix = []
        for i in range(self.hidden_dim):
            row = flat_w[i * self.vocab_size:(i + 1) * self.vocab_size]
            matrix.append(row)
        return matrix, bias

    def forward(self, tokens):
        """
        Forward pass: tokens → embedding → hidden → output → loss

        Returns: (loss, cache)
        cache stores intermediate values needed for backpropagation.
        """
        if len(tokens) < 2:
            return 0.0, None

        embedding = self._get_embedding()
        hidden_w, hidden_b = self._get_hidden()
        output_w, output_b = self._get_output()

        total_loss = 0.0
        count = 0
        cache = {
            'tokens': tokens,
            'embeds': [],
            'hiddens': [],
            'hidden_pre_relu': [],
            'logits_list': [],
        }

        # Process each pair of consecutive tokens (input → predict next)
        for t in range(len(tokens) - 1):
            input_token = tokens[t]
            target_token = tokens[t + 1]

            # Step 1: Embedding lookup
            embed = embedding[input_token][:]
            cache['embeds'].append(embed)

            # Step 2: Hidden layer — h = ReLU(embed @ hidden_w + hidden_b)
            hidden_pre = [0.0] * self.hidden_dim
            for j in range(self.hidden_dim):
                s = hidden_b[j]
                for k in range(self.embed_dim):
                    s += embed[k] * hidden_w[k][j]
                hidden_pre[j] = s

            cache['hidden_pre_relu'].append(hidden_pre[:])

            # ReLU activation: max(0, x)
            hidden = [max(0.0, x) for x in hidden_pre]
            cache['hiddens'].append(hidden)

            # Step 3: Output layer — logits = hidden @ output_w + output_b
            logits = [0.0] * self.vocab_size
            for j in range(self.vocab_size):
                s = output_b[j]
                for k in range(self.hidden_dim):
                    s += hidden[k] * output_w[k][j]
                logits[j] = s

            cache['logits_list'].append(logits)

            # Step 4: Softmax + Cross-entropy loss
            # Softmax converts logits to probabilities
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]

            # Cross-entropy loss: -log(probability of correct token)
            loss = -math.log(max(probs[target_token], 1e-10))
            total_loss += loss
            count += 1

        avg_loss = total_loss / max(count, 1)
        return avg_loss, cache

    def backward(self, cache):
        """
        Backward pass: compute gradients of loss with respect to all weights.

        This is BACKPROPAGATION — the core algorithm of deep learning.
        We compute d(loss)/d(weight) for every weight in the model.

        Returns: flat list of gradients (same length as self.weights)
        """
        tokens = cache['tokens']
        embedding = self._get_embedding()
        hidden_w, hidden_b = self._get_hidden()
        output_w, output_b = self._get_output()

        # Initialize gradient accumulators (zeros)
        grad_embedding = [[0.0] * self.embed_dim for _ in range(self.vocab_size)]
        grad_hidden_w = [[0.0] * self.hidden_dim for _ in range(self.embed_dim)]
        grad_hidden_b = [0.0] * self.hidden_dim
        grad_output_w = [[0.0] * self.vocab_size for _ in range(self.hidden_dim)]
        grad_output_b = [0.0] * self.vocab_size

        num_steps = len(tokens) - 1

        for t in range(num_steps):
            target_token = tokens[t + 1]
            input_token = tokens[t]
            embed = cache['embeds'][t]
            hidden = cache['hiddens'][t]
            hidden_pre = cache['hidden_pre_relu'][t]
            logits = cache['logits_list'][t]

            # ── Softmax probabilities ──
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]

            # ── Gradient of loss w.r.t. logits ──
            # d_loss/d_logits = probs - one_hot(target)
            d_logits = probs[:]
            d_logits[target_token] -= 1.0
            # Average over steps
            for j in range(self.vocab_size):
                d_logits[j] /= num_steps

            # ── Gradient of output layer ──
            # logits = hidden @ output_w + output_b
            for j in range(self.vocab_size):
                grad_output_b[j] += d_logits[j]
                for k in range(self.hidden_dim):
                    grad_output_w[k][j] += hidden[k] * d_logits[j]

            # ── Gradient flowing back to hidden ──
            d_hidden = [0.0] * self.hidden_dim
            for k in range(self.hidden_dim):
                for j in range(self.vocab_size):
                    d_hidden[k] += output_w[k][j] * d_logits[j]

            # ── ReLU gradient ──
            # d_relu/d_input = 1 if input > 0, else 0
            d_hidden_pre = [0.0] * self.hidden_dim
            for j in range(self.hidden_dim):
                if hidden_pre[j] > 0:
                    d_hidden_pre[j] = d_hidden[j]

            # ── Gradient of hidden layer ──
            # hidden_pre = embed @ hidden_w + hidden_b
            for j in range(self.hidden_dim):
                grad_hidden_b[j] += d_hidden_pre[j]
                for k in range(self.embed_dim):
                    grad_hidden_w[k][j] += embed[k] * d_hidden_pre[j]

            # ── Gradient of embedding ──
            d_embed = [0.0] * self.embed_dim
            for k in range(self.embed_dim):
                for j in range(self.hidden_dim):
                    d_embed[k] += hidden_w[k][j] * d_hidden_pre[j]

            for k in range(self.embed_dim):
                grad_embedding[input_token][k] += d_embed[k]

        # ── Flatten all gradients into a single vector ──
        # Must match the same order as self.weights
        gradients = []

        # Embedding gradients
        for i in range(self.vocab_size):
            gradients.extend(grad_embedding[i])

        # Hidden layer weight gradients
        for i in range(self.embed_dim):
            gradients.extend(grad_hidden_w[i])

        # Hidden bias gradients
        gradients.extend(grad_hidden_b)

        # Output layer weight gradients
        for i in range(self.hidden_dim):
            gradients.extend(grad_output_w[i])

        # Output bias gradients
        gradients.extend(grad_output_b)

        return gradients


# ──────────────────────────────────────────────────────────
# TRAINING FUNCTION — called by the Go worker
# ──────────────────────────────────────────────────────────

def train_on_shard(data_path, batch_size, learning_rate, weights_path=None):
    """
    Train the model on one data shard and return gradients.

    Args:
        data_path: Path to the text data file (e.g., dataset/shard1.txt)
        batch_size: Number of text lines to process per batch
        learning_rate: Not used here (master applies it), but logged
        weights_path: Path to JSON file with current model weights from master

    Returns:
        dict with 'gradients' (list of floats) and 'loss' (float)
    """
    # Load training data
    if not os.path.exists(data_path):
        return {"error": f"Data file not found: {data_path}", "gradients": [], "loss": 0.0}

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Create model
    model = SimpleModel(vocab_size=50, embed_dim=16, hidden_dim=32)

    # Load weights from master if provided
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            master_weights = json.load(f)
        if len(master_weights) == model.get_weight_count():
            model.load_weights(master_weights)

    # Tokenize all lines
    all_tokens = []
    for line in lines[:batch_size]:
        tokens = tokenize(line)
        if len(tokens) > 2:
            all_tokens.extend(tokens)

    if len(all_tokens) < 3:
        return {"error": "Not enough tokens to train", "gradients": [], "loss": 0.0}

    # Forward pass — compute loss
    loss, cache = model.forward(all_tokens)

    if cache is None:
        return {"error": "Forward pass failed", "gradients": [], "loss": 0.0}

    # Backward pass — compute gradients
    gradients = model.backward(cache)

    return {
        "gradients": gradients,
        "loss": round(loss, 6),
        "num_tokens": len(all_tokens),
        "num_parameters": model.get_weight_count(),
    }


# ──────────────────────────────────────────────────────────
# MAIN — entry point when called from Go worker
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LLM Training Worker Script")
    parser.add_argument("--data", required=True, help="Path to data shard file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weights", default="", help="Path to weights JSON file")
    args = parser.parse_args()

    # Print status to stderr (Go reads stdout for JSON only)
    print(f"[PYTHON] Training on: {args.data}", file=sys.stderr)
    print(f"[PYTHON] Batch size: {args.batch_size}, LR: {args.lr}", file=sys.stderr)

    result = train_on_shard(args.data, args.batch_size, args.lr, args.weights)

    print(f"[PYTHON] Loss: {result.get('loss', 'N/A')}", file=sys.stderr)
    print(f"[PYTHON] Tokens: {result.get('num_tokens', 0)}, Params: {result.get('num_parameters', 0)}", file=sys.stderr)

    # Output JSON to stdout — this is what Go reads
    json.dump(result, sys.stdout)
