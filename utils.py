import os
BASE_DIR = os.environ.get("MTKD_BASE_DIR", "/m/triton/scratch/elec/t405-puhe/p/bijoym1")
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def update_iemocap_label(label, column="emo"):
    if label == "ang":
        return "anger"
    elif label == "hap":
        return "happiness"
    elif label == "neu":
        return "neutral"
    elif label == "sad":
        return "sadness"

##########################################################################

def update_fesc_label(label, column="emo"):
    if label == "1":
        return "anger"
    elif label == "2":
        return "happiness"
    elif label == "3":
        return "neutral"
    elif label == "4":
        return "sadness"

##########################################################################

def is_common(emotion):
    if emotion == "neutral" or emotion == "sadness" or emotion == "happiness" or emotion == "anger":
        return 1
    else:
        return 0

########################################################################## 

def update_iemocap_path(path):
    return f"{BASE_DIR}/TempDatasets" + path.split("/m/teamwork/t40511_asr/c")[-1]

##########################################################################

# def update_cafe_path(path):
#     return "/scratch/elec/t405-puhe/p/bijoym1/TempDatasets" + path.split("/m/teamwork/t40511_asr/c/CaFE/data")[-1]

##########################################################################

def update_cafe_path(path):
    return "/content/CaFE/" + path

##########################################################################

# this fn is for Finnish dataset only
def update_file_path(path):
    return "/m/triton/scratch/elec/" + path.split("/m/triton/scratch/")[1]

##########################################################################

def update_audio_path(path):
    return f"{BASE_DIR}/TempDatasets" + path.split("/m/teamwork/t40511_asr/c")[-1]

##########################################################################

# this fn is for CaFE dataset only
def update_wav_path_cafe(path):
    return f"{BASE_DIR}/TempDatasets" + path.split("/m/teamwork/t40511_asr/c/CaFE/data")[-1]

##########################################################################

def to_label(emo, label2id):
    return label2id[emo]

##########################################################################

def preprocess_function(examples, feature_extractor):
    '''
    This function prepares the dataset for the transformer
    by applying the feature extractor to it (among other
    processes).
    '''
    max_duration = 20.0 # 20.0 # seconds
    # max_duration = 120.0 # seconds
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(audio_arrays,
                               sampling_rate=feature_extractor.sampling_rate,
                               max_length=int(feature_extractor.sampling_rate * max_duration),
                               padding=True,
                               truncation=True)
    return inputs

##########################################################################

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(example["input_values"]) for example in batch]
    labels = [int(example["label"]) for example in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    out = {"input_values": inputs, "label": torch.tensor(labels)}
    if "idx" in batch[0]:
        out["idx"] = torch.tensor([int(example["idx"]) for example in batch])
    return out

##########################################################################

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0

    # train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_OF_EPOCHS}, Training", leave=False)

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    for batch in train_loader:
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.logits

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loader.set_postfix(loss=loss.item())

        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    total_train_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, train_accuracy, total_train_loss, train_confidence_scores)

##########################################################################

def train_kd(teacher, student, train_loader, optimizer_ce, optimizer_kl, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher.train()
    student.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    # train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_OF_EPOCHS}, Training", leave=False)

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    for batch in train_loader:
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        optimizer_ce.zero_grad()
        optimizer_kl.zero_grad()

        teacher_outputs = teacher(inputs)
        teacher_outputs = teacher_outputs.logits
        teacher_outputs_soft = F.softmax(teacher_outputs / temperature, dim=-1)

        student_outputs = student(inputs)
        student_outputs = student_outputs.logits
        # student_outputs_soft = F.softmax(student_outputs / temperature, dim=-1)
        student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)

        loss_kl = loss_fn_kl(student_outputs_soft, teacher_outputs_soft) * (temperature**2)
        loss_ce = loss_fn_ce(student_outputs, labels)

        loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl

        loss.backward()

        optimizer_ce.step()

        train_loader.set_postfix(loss=loss.item())

        _, predicted = torch.max(student_outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()
        total_kl_loss += loss_kl.item()
        total_ce_loss += loss_ce.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    total_train_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)
    total_ce_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, train_accuracy, total_train_loss, total_kl_loss, total_ce_loss, train_confidence_scores)

##########################################################################

def calculate_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.view(tensor1.size(0), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), -1)
    cosine_sim = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    return cosine_sim.mean()

##########################################################################

def contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Computes contrastive loss using student embeddings and labels.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Normalize the embeddings to have unit L2 norm
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Subtract max for numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Mask out self-contrast (diagonal elements)
    logits_mask = torch.scatter(
        torch.ones_like(logits),
        1,
        torch.arange(batch_size, device=device).view(-1, 1),
        0
    )

    # Find positive mask: elements with same label, excluding self-contrast
    labels_col = labels.view(-1, 1)
    mask = torch.eq(labels_col, labels_col.T).float().to(device)
    mask = mask * logits_mask

    # Compute log-probabilities
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of log-likelihood over positive pairs
    num_positives = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(num_positives, min=1.0)

    # Negative of mean log-likelihood
    loss = -mean_log_prob_pos
    
    # Only compute loss for samples that actually have positive pairs in the batch
    valid_samples_mask = (num_positives > 0).float()
    if valid_samples_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    loss = (loss * valid_samples_mask).sum() / valid_samples_mask.sum()
    return loss

##########################################################################

@torch.no_grad()
def precompute_teacher_logits(teacher_en, teacher_fi, teacher_fr, loader, device,
                               dtype=torch.float16):
    """
    Runs each (frozen) teacher exactly ONCE over a loader and caches their
    logits in dataset order, in fp16 to keep the cache small.

    Why this matters for low-resource / low-memory training:
    Teachers never change after this point (they are frozen), so re-running
    three full Wav2Vec2 forward passes on every single training step is
    pure waste -- it costs the same GPU memory and compute every epoch for
    an output that never changes. Caching once turns every later epoch's
    per-step cost from "4 forward passes + 1 backward pass" into
    "1 forward pass + 1 backward pass" (student only), and after caching,
    the teacher models can be moved off the GPU entirely (see pipeline.py).

    Requires the dataset to expose a stable per-sample 'idx' field (see
    main.py / data.py changes) so cached rows can be looked up by index
    regardless of shuffling in the training DataLoader.

    Returns a dict: {'en': Tensor[N, K], 'fi': Tensor[N, K], 'fr': Tensor[N, K]}
    """
    teacher_en.eval()
    teacher_fi.eval()
    teacher_fr.eval()

    en_logits, fi_logits, fr_logits, indices = [], [], [], []

    for batch in loader:
        inputs = batch['input_values'].to(device)
        idx = batch['idx']

        en_logits.append(teacher_en(inputs).logits.to(dtype).cpu())
        fi_logits.append(teacher_fi(inputs).logits.to(dtype).cpu())
        fr_logits.append(teacher_fr(inputs).logits.to(dtype).cpu())
        indices.append(idx.cpu())

    indices = torch.cat(indices, dim=0)
    n = int(indices.max().item()) + 1
    num_classes = torch.cat(en_logits, dim=0).shape[-1]

    def scatter_by_idx(chunks):
        full = torch.zeros((n, num_classes), dtype=dtype)
        full[indices] = torch.cat(chunks, dim=0)
        return full

    cache = {
        'en': scatter_by_idx(en_logits),
        'fi': scatter_by_idx(fi_logits),
        'fr': scatter_by_idx(fr_logits),
    }
    return cache

##########################################################################

class EmbeddingMemoryBank:
    """
    Small FIFO queue of (embedding, label) pairs from recent batches, used
    to give the supervised contrastive loss a larger effective pool of
    positives/negatives than the *physical* batch size allows.

    On low-memory hardware you may be forced to use a small batch size
    (e.g. 8), which often leaves too few same-class samples in a batch for
    SCL to learn from. Concatenating a queue of detached embeddings from
    recent steps (MoCo-style) restores a usefully large contrastive pool
    without increasing activation memory, since queued embeddings carry no
    gradient and no autograd graph.
    """
    def __init__(self, max_size=512, embedding_dim=None, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.embeddings = torch.empty(0, embedding_dim) if embedding_dim else None
        self.labels = torch.empty(0, dtype=torch.long)

    def push(self, embeddings, labels):
        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()
        if self.embeddings is None or self.embeddings.numel() == 0:
            self.embeddings, self.labels = embeddings, labels
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)
        if self.embeddings.size(0) > self.max_size:
            self.embeddings = self.embeddings[-self.max_size:]
            self.labels = self.labels[-self.max_size:]

    def get(self, device):
        return self.embeddings.to(device), self.labels.to(device)

##########################################################################

def contrastive_loss_with_bank(embeddings, labels, memory_bank=None, temperature=0.07):
    """
    Same math as `contrastive_loss`, but optionally extends the negative/
    positive pool with a memory bank of embeddings from recent batches
    before computing the loss, then pushes the current batch into the bank.
    Use this instead of `contrastive_loss` when batch size must be kept
    small for memory reasons.
    """
    if memory_bank is not None:
        bank_embeddings, bank_labels = memory_bank.get(embeddings.device)
        if bank_embeddings.numel() > 0:
            all_embeddings = torch.cat([embeddings, bank_embeddings], dim=0)
            all_labels = torch.cat([labels, bank_labels], dim=0)
        else:
            all_embeddings, all_labels = embeddings, labels
        memory_bank.push(embeddings, labels)
    else:
        all_embeddings, all_labels = embeddings, labels

    return contrastive_loss(all_embeddings, all_labels, temperature=temperature)

##########################################################################

def train_mtkd(
    teacher_en, teacher_fi, teacher_fr, student, train_loader,
    optimizer, loss_fn_ce, loss_fn_kl, device,
    temperature=5, lambda_param=0.25,
    attention_net=None, teacher_selection="cosine",
    ce_weight=1.0, kd_weight=1.0, contrastive_weight=0.0,
    contrastive_temp=0.07,
    cached_teacher_logits=None, use_amp=False, scaler=None,
    grad_accum_steps=1
):
    # Teachers are frozen -> they must be in eval() mode (disables dropout),
    # never train() mode. Running frozen teachers in train() mode was a bug:
    # it injected dropout noise into "ground truth" teacher targets every step.
    if teacher_en is not None:
        teacher_en.eval()
    if teacher_fi is not None:
        teacher_fi.eval()
    if teacher_fr is not None:
        teacher_fr.eval()
    student.train()
    if attention_net is not None:
        attention_net.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0
    total_contrastive_loss = 0

    total_weight_en = 0.0
    total_weight_fi = 0.0
    total_weight_fr = 0.0

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        # ---- Teacher outputs: either looked up from a precomputed cache
        # (no teacher forward pass at all -> the single biggest memory/compute
        # saving available, since teachers are frozen and never change), or
        # computed on the fly under no_grad if no cache exists.
        if cached_teacher_logits is not None:
            idx = batch['idx']
            teacher_en_outputs = cached_teacher_logits['en'][idx].to(device).float()
            teacher_fi_outputs = cached_teacher_logits['fi'][idx].to(device).float()
            teacher_fr_outputs = cached_teacher_logits['fr'][idx].to(device).float()
        else:
            with torch.no_grad():
                teacher_en_outputs = teacher_en(inputs).logits
                teacher_fi_outputs = teacher_fi(inputs).logits
                teacher_fr_outputs = teacher_fr(inputs).logits

        with torch.autocast(device_type=device.type, enabled=use_amp):
            # Forward student and extract logits and last layer hidden states for embeddings
            student_model_outputs = student(inputs, output_hidden_states=True)
            student_outputs = student_model_outputs.logits
            student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)

            # Extract student embeddings from last hidden state
            last_hidden_states = student_model_outputs.hidden_states[-1]
            mean_pooled = last_hidden_states.mean(dim=1)
            if hasattr(student, "projector") and student.projector is not None:
                student_embeddings = student.projector(mean_pooled)
            else:
                student_embeddings = mean_pooled

            if teacher_selection == "cosine":
                cos_sim_en = calculate_cosine_similarity(student_outputs, teacher_en_outputs)
                cos_sim_fi = calculate_cosine_similarity(student_outputs, teacher_fi_outputs)
                cos_sim_fr = calculate_cosine_similarity(student_outputs, teacher_fr_outputs)

                cosine_sims = torch.tensor([cos_sim_en, cos_sim_fi, cos_sim_fr], device=device)
                weights = F.softmax(cosine_sims / 0.25, dim=-1)
                weight_en, weight_fi, weight_fr = weights
                
                # Use raw logits for cosine (must be homogeneous)
                teacher_en_outputs_soft = F.softmax(teacher_en_outputs / temperature, dim=-1)
                teacher_fi_outputs_soft = F.softmax(teacher_fi_outputs / temperature, dim=-1)
                teacher_fr_outputs_soft = F.softmax(teacher_fr_outputs / temperature, dim=-1)
            elif teacher_selection == "attention":
                # Trainable attention network outputs weights and projected teacher logits
                attention_weights, projected_teachers = attention_net(student_outputs, [teacher_en_outputs, teacher_fi_outputs, teacher_fr_outputs])
                # Average over batch to get [3] weights for KL loss weighting
                mean_weights = attention_weights.mean(dim=0)
                weight_en, weight_fi, weight_fr = mean_weights
                
                # Use projected logits for attention (supports heterogeneous teachers)
                teacher_en_outputs_soft = F.softmax(projected_teachers[0] / temperature, dim=-1)
                teacher_fi_outputs_soft = F.softmax(projected_teachers[1] / temperature, dim=-1)
                teacher_fr_outputs_soft = F.softmax(projected_teachers[2] / temperature, dim=-1)
            else:
                raise ValueError(f"Unknown teacher selection mechanism: {teacher_selection}")

            total_weight_en += weight_en.item()
            total_weight_fi += weight_fi.item()
            total_weight_fr += weight_fr.item()

            loss_teacher_en = loss_fn_kl(student_outputs_soft, teacher_en_outputs_soft) * (temperature**2)
            loss_teacher_fi = loss_fn_kl(student_outputs_soft, teacher_fi_outputs_soft) * (temperature**2)
            loss_teacher_fr = loss_fn_kl(student_outputs_soft, teacher_fr_outputs_soft) * (temperature**2)

            loss_ce = loss_fn_ce(student_outputs, labels)
            loss_kl = weight_en * loss_teacher_en + weight_fi * loss_teacher_fi + weight_fr * loss_teacher_fr
            loss_contrastive = contrastive_loss(student_embeddings, labels, temperature=contrastive_temp)

            # Final loss should be: ce_weight * CE_loss + kd_weight * KL_loss + contrastive_weight * Contrastive_loss
            loss = ce_weight * loss_ce + kd_weight * loss_kl + contrastive_weight * loss_contrastive
            # Normalize by grad_accum_steps so accumulated gradients equal the
            # average over the full effective batch, not the sum.
            loss_to_backprop = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        if (step + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loader.set_postfix(
            loss=loss.item(),
            ce=loss_ce.item(),
            kl=loss_kl.item(),
            scl=loss_contrastive.item()
        )

        _, predicted = torch.max(student_outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()
        total_kl_loss += loss_kl.item()
        total_ce_loss += loss_ce.item()
        total_contrastive_loss += loss_contrastive.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    # Flush any leftover gradients from a final partial accumulation cycle.
    if (step + 1) % grad_accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    total_train_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)
    total_ce_loss /= len(train_loader)
    total_contrastive_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    avg_attention_weights = [
        total_weight_en / len(train_loader),
        total_weight_fi / len(train_loader),
        total_weight_fr / len(train_loader)
    ]

    return (
        unweighted_recall, weighted_recall, train_accuracy,
        total_train_loss, total_kl_loss, total_ce_loss, total_contrastive_loss,
        train_confidence_scores, avg_attention_weights
    )

##########################################################################

def validation_v2(model, valid_loader, loss_fn, device):
    model.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.logits

            loss = loss_fn(outputs, labels)

            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation(model, valid_loader, loss_fn, device):
    model.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.logits

            loss = loss_fn(outputs, labels)

            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation_kd(teacher, student, valid_loader, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher.eval()
    student.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            teacher_outputs = teacher(inputs)
            teacher_outputs = teacher_outputs.logits
            teacher_outputs_soft = F.softmax(teacher_outputs / temperature, dim=-1)

            student_outputs = student(inputs)
            student_outputs = student_outputs.logits
            # student_outputs_soft = F.softmax(student_outputs / temperature, dim=-1)
            student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)

            loss_kl = loss_fn_kl(student_outputs_soft, teacher_outputs_soft) * (temperature**2)
            loss_ce = loss_fn_ce(student_outputs, labels)

            loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl


            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(student_outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()
            total_kl_loss += loss_kl.item()
            total_ce_loss += loss_ce.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        total_kl_loss /= len(valid_loader)
        total_ce_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, total_kl_loss, total_ce_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation_mtkd(
    teacher_en, teacher_fi, teacher_fr, student, valid_loader, 
    loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25,
    attention_net=None, teacher_selection="cosine",
    ce_weight=1.0, kd_weight=1.0, contrastive_weight=0.0, 
    contrastive_temp=0.07, cached_teacher_logits=None
):
    if teacher_en is not None:
        teacher_en.eval()
    if teacher_fi is not None:
        teacher_fi.eval()
    if teacher_fr is not None:
        teacher_fr.eval()
    student.eval()
    if attention_net is not None:
        attention_net.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0
    total_contrastive_loss = 0

    total_weight_en = 0.0
    total_weight_fi = 0.0
    total_weight_fr = 0.0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    epoch_attention_weights = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            if cached_teacher_logits is not None:
                idx = batch['idx']
                teacher_en_outputs = cached_teacher_logits['en'][idx].to(device).float()
                teacher_fi_outputs = cached_teacher_logits['fi'][idx].to(device).float()
                teacher_fr_outputs = cached_teacher_logits['fr'][idx].to(device).float()
            else:
                teacher_en_outputs = teacher_en(inputs).logits
                teacher_fi_outputs = teacher_fi(inputs).logits
                teacher_fr_outputs = teacher_fr(inputs).logits

            # Forward student and extract logits and last layer hidden states for embeddings
            student_model_outputs = student(inputs, output_hidden_states=True)
            student_outputs = student_model_outputs.logits
            student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)
            
            # Extract student embeddings from last hidden state
            last_hidden_states = student_model_outputs.hidden_states[-1]
            mean_pooled = last_hidden_states.mean(dim=1)
            if hasattr(student, "projector") and student.projector is not None:
                student_embeddings = student.projector(mean_pooled)
            else:
                student_embeddings = mean_pooled

            if teacher_selection == "cosine":
                cos_sim_en = calculate_cosine_similarity(student_outputs, teacher_en_outputs)
                cos_sim_fi = calculate_cosine_similarity(student_outputs, teacher_fi_outputs)
                cos_sim_fr = calculate_cosine_similarity(student_outputs, teacher_fr_outputs)

                cosine_sims = torch.tensor([cos_sim_en, cos_sim_fi, cos_sim_fr], device=device)
                weights = F.softmax(cosine_sims / 0.25, dim=-1)
                weight_en, weight_fi, weight_fr = weights
                
                # Use raw logits for cosine
                teacher_en_outputs_soft = F.softmax(teacher_en_outputs / temperature, dim=-1)
                teacher_fi_outputs_soft = F.softmax(teacher_fi_outputs / temperature, dim=-1)
                teacher_fr_outputs_soft = F.softmax(teacher_fr_outputs / temperature, dim=-1)
                
                # Save batch weights duplicated for each sample
                batch_weights = weights.unsqueeze(0).repeat(inputs.size(0), 1)
                epoch_attention_weights.append(batch_weights.cpu())
            elif teacher_selection == "attention":
                # Trainable attention network outputs weights and projected teachers
                attention_weights, projected_teachers = attention_net(student_outputs, [teacher_en_outputs, teacher_fi_outputs, teacher_fr_outputs])
                # Save sample-wise weights
                epoch_attention_weights.append(attention_weights.detach().cpu())
                
                # For computation of KL loss in this batch, we average over the batch
                mean_weights = attention_weights.mean(dim=0)
                weight_en, weight_fi, weight_fr = mean_weights
                
                # Use projected logits for attention
                teacher_en_outputs_soft = F.softmax(projected_teachers[0] / temperature, dim=-1)
                teacher_fi_outputs_soft = F.softmax(projected_teachers[1] / temperature, dim=-1)
                teacher_fr_outputs_soft = F.softmax(projected_teachers[2] / temperature, dim=-1)
            else:
                raise ValueError(f"Unknown teacher selection mechanism: {teacher_selection}")

            total_weight_en += weight_en.item()
            total_weight_fi += weight_fi.item()
            total_weight_fr += weight_fr.item()

            loss_teacher_en = loss_fn_kl(student_outputs_soft, teacher_en_outputs_soft) * (temperature**2)
            loss_teacher_fi = loss_fn_kl(student_outputs_soft, teacher_fi_outputs_soft) * (temperature**2)
            loss_teacher_fr = loss_fn_kl(student_outputs_soft, teacher_fr_outputs_soft) * (temperature**2)

            loss_ce = loss_fn_ce(student_outputs, labels)
            loss_kl = weight_en * loss_teacher_en + weight_fi * loss_teacher_fi + weight_fr * loss_teacher_fr
            loss_contrastive = contrastive_loss(student_embeddings, labels, temperature=contrastive_temp)

            # Final loss should be: ce_weight * CE_loss + kd_weight * KL_loss + contrastive_weight * Contrastive_loss
            loss = ce_weight * loss_ce + kd_weight * loss_kl + contrastive_weight * loss_contrastive

            valid_loader.set_postfix(
                loss=loss.item(),
                ce=loss_ce.item(),
                kl=loss_kl.item(),
                scl=loss_contrastive.item()
            )

            _, predicted = torch.max(student_outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()
            total_kl_loss += loss_kl.item()
            total_ce_loss += loss_ce.item()
            total_contrastive_loss += loss_contrastive.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        total_kl_loss /= len(valid_loader)
        total_ce_loss /= len(valid_loader)
        total_contrastive_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    avg_attention_weights = [
        total_weight_en / len(valid_loader),
        total_weight_fi / len(valid_loader),
        total_weight_fr / len(valid_loader)
    ]

    if len(epoch_attention_weights) > 0:
        epoch_attention_weights = torch.cat(epoch_attention_weights, dim=0)
    else:
        epoch_attention_weights = torch.empty(0)

    return (
        unweighted_recall, weighted_recall, valid_accuracy, 
        total_valid_loss, total_kl_loss, total_ce_loss, total_contrastive_loss, 
        valid_confidence_scores, all_actual_labels, all_predictions,
        avg_attention_weights, epoch_attention_weights
    )

##########################################################################

def keep_first_n(data, n=10):
    class_count = {}
    filtered_data = []

    for item in data:
        label = item[1]
        
        if label not in class_count:
            class_count[label] = 0

        if class_count[label] < n:
            filtered_data.append(item)
            class_count[label] += 1

    return filtered_data

##########################################################################

def plot_confidence_scores(data, title, file_id):
    data = sorted(data, key=lambda x: x[1])
    data = keep_first_n(data, n=12)
    probabilities, class_labels = zip(*data)
    plt.figure(figsize=(12, 5))
    plt.bar([x for x in range(len(class_labels))], probabilities, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Probability Score')
    plt.title(title)
    plt.xticks(range(len(data)), class_labels)
    filename = f"{BASE_DIR}/SER/FTWav2Vec2/figures/cs_{title.lower().split()[0]}_{file_id}.jpg"
    plt.savefig(filename, format='jpg')
    plt.show()

##########################################################################

def plot_confusion_matrix(actual, predicted, file_id):
    data = confusion_matrix(actual, predicted)
    num_classes = max(max(actual), max(predicted)) + 1  # Get the maximum class label present in actual and predicted lists
    df_cm = pd.DataFrame(data, range(num_classes), range(num_classes))

    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 12})

    plt.ylabel("Actual")
    plt.xlabel("Predicted")  # Corrected xlabel
    plt.title(f"Session {file_id}")

    filename = f"cm_{file_id}.jpg"
    plt.savefig(filename, format='jpg')
    plt.show()

##########################################################################
