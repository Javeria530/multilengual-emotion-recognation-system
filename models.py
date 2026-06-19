from transformers import AutoModelForAudioClassification
import torch
import os
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def model_ft(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"

    model = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    model.freeze_feature_encoder()
    model.to(device)
    return model

##########################################################################

def model_kd(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"

    teacher = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    file_path_teacher = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2_iemocap_multilingual.pth"
    print("START: Load Multilingual Teacher's Knowledge")
    if os.path.exists(file_path_teacher):
        checkpoint = torch.load(file_path_teacher)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Multilingual Teacher's Knowledge")
    teacher.freeze_feature_encoder()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)

    student = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    student.freeze_feature_encoder()
    student.to(device)
    return teacher, student

##########################################################################

def model_mtkd(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"
    teacher_en = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_en = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2.pth"
    print("START: Load Monolingual English Teacher's Knowledge")
    if os.path.exists(file_path_teacher_en):
        checkpoint = torch.load(file_path_teacher_en)
        teacher_en.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual English Teacher's Knowledge")
    teacher_en.freeze_feature_encoder()
    for param in teacher_en.parameters():
        param.requires_grad = False
    teacher_en.to(device)

    teacher_fi = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_fi = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_JAKA.pth"
    print("START: Load Monolingual Finnish Teacher's Knowledge")
    if os.path.exists(file_path_teacher_fi):
        checkpoint = torch.load(file_path_teacher_fi)
        teacher_fi.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual Finnish Teacher's Knowledge")
    teacher_fi.freeze_feature_encoder()
    for param in teacher_fi.parameters():
        param.requires_grad = False
    teacher_fi.to(device)

    teacher_fr = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_fr = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_CaFE.pth"
    print("START: Load Monolingual French Teacher's Knowledge")
    if os.path.exists(file_path_teacher_fr):
        checkpoint = torch.load(file_path_teacher_fr)
        teacher_fr.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual French Teacher's Knowledge")
    teacher_fr.freeze_feature_encoder()
    for param in teacher_fr.parameters():
        param.requires_grad = False
    teacher_fr.to(device)

    student = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    student.freeze_feature_encoder()
    student.to(device)

    return teacher_en, teacher_fi, teacher_fr, student

##########################################################################

class AttentionMTKD(torch.nn.Module):
    """
    AttentionMTKD: A trainable attention mechanism for Multi-Teacher Knowledge Distillation.
    
    It learns dynamic attention weights to weigh the KL divergence losses of multiple 
    teacher models based on student and teacher logits.
    
    Architecture:
        Linear(in_features = (num_teachers + 1) * num_classes, out_features = hidden_dim)
        -> ReLU()
        -> Linear(in_features = hidden_dim, out_features = num_teachers)
        -> Softmax(dim=-1)
    
    Output:
        attention_weights: Tensor of shape [batch_size, num_teachers] summing to 1.0 along the last dimension.
    """
    def __init__(self, num_classes=4, hidden_dim=16, num_teachers=3):
        super(AttentionMTKD, self).__init__()
        # Input features: concatenated student and teacher logits
        input_dim = (num_teachers + 1) * num_classes
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_teachers)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, student_logits, teacher_logits_list):
        # Concatenate student logits and all teacher logits along last dimension
        # student_logits shape: [batch_size, num_classes]
        # each teacher logit shape: [batch_size, num_classes]
        # Output concatenated shape: [batch_size, (num_teachers + 1) * num_classes]
        all_logits = [student_logits] + teacher_logits_list
        x = torch.cat(all_logits, dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights

##########################################################################
