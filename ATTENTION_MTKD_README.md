# AttentionMTKD & Supervised Contrastive Learning (SCL) for SER

This document explains the architecture, mathematical formulation, code modifications, and usage guidelines for the **AttentionMTKD** teacher weighting network and the **Supervised Contrastive Learning (SCL)** loss in this repository.

---

## 1. AttentionMTKD Architecture

### Overview
In standard Multi-Teacher Knowledge Distillation (MTKD), the importance of each teacher model's knowledge is computed using the cosine similarity between the student's output logits and each teacher's output logits.
`AttentionMTKD` replaces this static cosine similarity heuristic with a trainable, sample-specific attention network that dynamically assigns importance weights to each teacher based on their logit outputs.

### Architecture Details
The trainable attention network is a Feed-Forward Neural Network (FFN) with the following structure:
1. **Input Concatenation**:
   The network receives student logits ($S$) and all teacher logits ($T_1, T_2, T_3$). For $K$ classes and $N$ teachers, they are concatenated along the feature dimension to form the input vector:
   $$X = [S \mathbin{\Vert} T_1 \mathbin{\Vert} T_2 \mathbin{\Vert} T_3] \in \mathbb{R}^{B \times (N+1)K}$$
   where $B$ is the batch size, and $K$ is the number of emotion classes (4 for IEMOCAP, FESC, and CaFE).
2. **Hidden Layer**:
   A linear projection followed by a Rectified Linear Unit (ReLU) activation:
   $$H = \text{ReLU}(X W_1 + b_1)$$
   where $W_1 \in \mathbb{R}^{(N+1)K \times D}$ (with hidden dimension $D = 16$) and $b_1 \in \mathbb{R}^D$.
3. **Output Layer**:
   A second linear layer followed by a Softmax activation to normalize teacher weights across all teachers:
   $$W = \text{Softmax}(H W_2 + b_2)$$
   where $W_2 \in \mathbb{R}^{D \times N}$, $b_2 \in \mathbb{R}^N$, and $W \in \mathbb{R}^{B \times N}$ represents the sample-wise attention weights.
4. **Attention Weights**:
   The output weights for teacher $i$ are denoted as:
   $$\mathbf{attention\_weights} = [w_1, w_2, w_3], \quad \sum_{i=1}^3 w_i = 1$$

---

## 2. Mathematical Formulation

### A. Attention-Weighted KL Divergence Loss
Instead of using cosine similarity weights, the total KL Divergence loss ($L_{KD}$) is weighted using the trainable attention weights $w_i$.
For a batch $B$:
$$L_{KD} = \frac{1}{|B|} \sum_{b=1}^{|B|} \sum_{i=1}^3 w_{i, b} \cdot D_{KL}\left( \sigma(\mathbf{z}_{student, b} / \tau) \mathbin{\Vert} \sigma(\mathbf{z}_{teacher\_i, b} / \tau) \right) \cdot \tau^2$$
where $\tau$ is the temperature parameter, and $\sigma$ represents the softmax operation.

### B. Supervised Contrastive Learning (SCL) Loss
Supervised Contrastive Learning (SCL) encourages student representations of the same emotion class to cluster closely in the embedding space while pushing representations of different emotion classes apart.

For a batch of student embeddings $\{z_1, z_2, \dots, z_B\}$, the SCL loss is computed as:
$$L_{SCL} = \sum_{i=1}^{B} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau_{scl})}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau_{scl})}$$
where:
- $z_i = \text{Normalize}(E_i) \in \mathbb{R}^M$ is the L2-normalized sequence-level embedding of sample $i$.
- $E_i$ is the student embedding extracted from the Wav2Vec2 encoder's last layer hidden states after mean pooling and passing through the student's projection head.
- $A(i) = B \setminus \{i\}$ is the set of all other samples in the batch.
- $P(i) = \{p \in A(i) \mid y_p = y_i\}$ is the set of all positive samples sharing the same emotion label $y_i$ as sample $i$.
- $\tau_{scl}$ is the contrastive temperature parameter (default: $0.07$).

### C. Final Combined Loss
The total objective function optimized during distillation is:
$$L_{total} = \lambda_{ce} \cdot L_{CE} + \lambda_{kd} \cdot L_{KD} + \lambda_{scl} \cdot L_{SCL}$$
where $\lambda_{ce}$, $\lambda_{kd}$, and $\lambda_{scl}$ are configurable weights.

---

## 3. Code Modifications

A summary of file additions and changes:

1. **[config.json](file:///e:/Research%20Work/mtkd4ser/config.json)** [NEW]: Configures default weights for $L_{CE}$, $L_{KD}$, $L_{SCL}$, and hyperparameters for the attention hidden dimension and SCL temperature.
2. **[models.py](file:///e:/Research%20Work/mtkd4ser/models.py)** [MODIFY]: Adds the `AttentionMTKD` PyTorch module.
3. **[utils.py](file:///e:/Research%20Work/mtkd4ser/utils.py)** [MODIFY]:
   - Implements `contrastive_loss()` utility.
   - Comments out the original cosine similarity block in `train_mtkd` and `validation_mtkd` and replaces it with dynamic routing supporting both cosine and trainable attention selection.
   - Extracts student embeddings from Wav2Vec2 sequence hidden states for SCL computation.
   - Enriches return structures to include individual loss values and attention weights.
4. **[pipeline.py](file:///e:/Research%20Work/mtkd4ser/pipeline.py)** [MODIFY]:
   - Integrates the `AttentionMTKD` network into training and validation pipelines.
   - Optimizes student parameters along with `AttentionMTKD` parameters.
   - Logs CE, KL, Contrastive, and Total losses, plus average teacher weights to TensorBoard.
   - Saves validation attention weights for visualization.
5. **[main.py](file:///e:/Research%20Work/mtkd4ser/main.py)** [MODIFY]: Adds CLI flags to parse selection mechanisms (`--teacher_selection`), configuration path (`--config_path`), and allows loss weight overrides.

---

## 4. Training and Evaluation Commands

### Train with Trainable Attention & Supervised Contrastive Learning (AttentionMTKD)
To train the student using the trainable attention mechanism and SCL loss:
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 1 --TRAINING 1 --PARADIGM "MTKD" --LANGUAGE "EN" --LINGUALITY "Monolingual" --teacher_selection attention --config_path config.json
```

### Reproduce Original Paper Results (Cosine Selection, No SCL)
To maintain backward compatibility and run the original cosine-similarity-based teacher weighting without SCL loss:
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 1 --TRAINING 1 --PARADIGM "MTKD" --LANGUAGE "EN" --LINGUALITY "Monolingual" --teacher_selection cosine --ce_weight 0.75 --kd_weight 0.25 --contrastive_weight 0.0
```

### Run Evaluation (Validation / Test Mode)
To evaluate an existing checkpoint and save the attention weights for visualization:
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 1 --TRAINING 0 --PARADIGM "MTKD" --LANGUAGE "EN" --LINGUALITY "Monolingual" --teacher_selection attention --config_path config.json
```
Validation attention weights will be saved as a PyTorch tensor `.pt` file under `checkpoints/mtkd/visualizations/attention_weights_Monolingual_EN_S1_epochX.pt`.
