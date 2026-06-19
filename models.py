import os
BASE_DIR = os.environ.get("MTKD_BASE_DIR", "/m/triton/scratch/elec/t405-puhe/p/bijoym1")
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
    file_path_teacher = f"{BASE_DIR}/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2_iemocap_multilingual.pth"
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
    file_path_teacher_en = f"{BASE_DIR}/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2.pth"
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
    file_path_teacher_fi = f"{BASE_DIR}/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_JAKA.pth"
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
    file_path_teacher_fr = f"{BASE_DIR}/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_CaFE.pth"
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

class TeacherProjector(torch.nn.Module):
    """
    Per-teacher adapter that maps a teacher's native output (its own logit
    dimensionality / label space) into a shared space of size `shared_dim`.

    This is what lets AttentionMTKD work with HETEROGENEOUS teachers: a
    teacher does not need the same number of classes, or even the same
    architecture, as the student or the other teachers, as long as it
    produces a fixed-size vector per sample. A separate, tiny, learnable
    projector is created per teacher.
    """
    def __init__(self, teacher_output_dim, shared_dim):
        super(TeacherProjector, self).__init__()
        self.proj = torch.nn.Linear(teacher_output_dim, shared_dim)

    def forward(self, teacher_output):
        return self.proj(teacher_output)

##########################################################################

class AttentionMTKD(torch.nn.Module):
    """
    AttentionMTKD: A trainable attention mechanism for Multi-Teacher Knowledge Distillation.

    It learns dynamic, sample-specific attention weights to weigh the KL
    divergence losses of multiple teacher models, based on student and
    (projected) teacher outputs.

    Heterogeneous-teacher support: pass `teacher_dims`, a list with one
    entry per teacher giving that teacher's native output dimensionality.
    Each teacher gets its own `TeacherProjector` into a shared `proj_dim`
    space before the attention computation, so teachers no longer need
    identical architectures or label spaces. If `teacher_dims` is omitted,
    all teachers are assumed homogeneous with `num_classes` outputs
    (original behaviour, fully backward compatible).

    Output:
        attention_weights: Tensor of shape [batch_size, num_teachers] summing to 1.0 along the last dimension.
    """
    def __init__(self, num_classes=4, hidden_dim=16, num_teachers=3,
                 teacher_dims=None, proj_dim=None):
        super(AttentionMTKD, self).__init__()
        self.num_teachers = num_teachers
        self.proj_dim = proj_dim if proj_dim is not None else num_classes

        if teacher_dims is None:
            # Homogeneous case: every teacher already outputs num_classes logits.
            teacher_dims = [num_classes] * num_teachers

        self.teacher_projectors = torch.nn.ModuleList([
            TeacherProjector(dim, self.proj_dim) for dim in teacher_dims
        ])

        input_dim = num_classes + num_teachers * self.proj_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_teachers)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, student_logits, teacher_logits_list):
        # Each teacher's native output is projected into the shared proj_dim space first.
        projected = [
            proj(t_out) for proj, t_out in zip(self.teacher_projectors, teacher_logits_list)
        ]
        x = torch.cat([student_logits] + projected, dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights, projected

##########################################################################
