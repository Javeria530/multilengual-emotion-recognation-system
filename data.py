import os
BASE_DIR = os.environ.get("MTKD_BASE_DIR", "/m/triton/scratch/elec/t405-puhe/p/bijoym1")

from utils import (
    update_iemocap_path,
    update_file_path,
    to_label,
    is_common,
    update_cafe_path,
    update_iemocap_label, 
    update_fesc_label
)
import json
import pandas as pd
from datasets import DatasetDict, Audio, Dataset
import glob
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def iemocap(session):
    train_json_file = f"{BASE_DIR}/iemocap/session{session}/train.json"
    test_json_file = f"{BASE_DIR}/iemocap/session{session}/test.json"

    with open(train_json_file, "r") as f:
        train_data = json.load(f)
    with open(test_json_file, "r") as f:
        test_data = json.load(f)
    
    train_df = pd.DataFrame.from_dict(train_data, orient="index").reset_index()
    train_df = train_df.rename(columns={'index': 'file_id', 'wav': 'audio'})

    test_df = pd.DataFrame.from_dict(test_data, orient="index").reset_index()
    test_df = test_df.rename(columns={'index': 'file_id', 'wav': 'audio'})

    train_df["audio"] = train_df["audio"].apply(update_iemocap_path).values
    test_df["audio"] = test_df["audio"].apply(update_iemocap_path).values

    train_df["emo"] = train_df["emo"].apply(update_iemocap_label)
    test_df["emo"] = test_df["emo"].apply(update_iemocap_label)

    labels = ['anger', 'happiness', 'neutral', 'sadness'] #["ang", "hap", "neu", "sad"]
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # convert categorical label to numerical labels
    train_df["label"] = train_df["emo"].apply(to_label, args=(label2id,))
    test_df["label"] = test_df["emo"].apply(to_label, args=(label2id,))

    train_audio_data = Dataset.from_pandas(train_df[['audio', 'label']])
    train_audio_data = train_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    test_audio_data = Dataset.from_pandas(test_df[['audio', 'label']])
    test_audio_data = test_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        'train' : train_audio_data,
        'test' : test_audio_data,
        'dev' : test_audio_data,
    })

    return label2id, id2label, ds

##########################################################################

def low_resource_lang(train_manifest, test_manifest, label_map=None, sampling_rate=16_000):
    """
    Generic loader for a NEW low-resource language, kept separate from the
    EN/FI/FR loaders above so adding a language never requires touching
    those again.

    Expects two manifest files (csv or tsv, comma- or tab-separated -- auto
    detected), each with at minimum the columns:
        audio  : path to a .wav file
        emo    : the raw emotion label as it appears in your corpus

    `label_map` lets you map your corpus's raw emotion strings onto the
    same 4-class taxonomy (anger/happiness/neutral/sadness) the EN/FI/FR
    teachers already use -- e.g. {"ghussa": "anger", "khushi": "happiness", ...}.
    This keeps the new language plug-compatible with the existing teachers
    and AttentionMTKD setup with zero architecture changes.

    If your corpus's emotion taxonomy genuinely doesn't map cleanly onto
    these 4 classes (e.g. it has additional categories like "fear" or
    "disgust"), don't force a mapping here -- instead route this language's
    student head through a separate classifier/projector and the
    heterogeneous-teacher path in AttentionMTKD (see models.py
    TeacherProjector), so labels aren't silently collapsed into the wrong
    bucket.

    Returns (label2id, id2label, ds) with the same shape as iemocap()/
    fesc()/cafe(), so it works with the rest of the pipeline unmodified.
    """
    def _read_manifest(path):
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        if label_map is not None:
            df["emo"] = df["emo"].map(label_map)
        df = df.dropna(subset=["emo"])
        return df

    train_df = _read_manifest(train_manifest)
    test_df = _read_manifest(test_manifest)

    labels = ['anger', 'happiness', 'neutral', 'sadness']
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    train_df["label"] = train_df["emo"].apply(to_label, args=(label2id,))
    test_df["label"] = test_df["emo"].apply(to_label, args=(label2id,))

    train_audio_data = Dataset.from_pandas(train_df[['audio', 'label']])
    train_audio_data = train_audio_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    test_audio_data = Dataset.from_pandas(test_df[['audio', 'label']])
    test_audio_data = test_audio_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    ds = DatasetDict({
        'train': train_audio_data,
        'test': test_audio_data,
        'dev': test_audio_data,
    })

    return label2id, id2label, ds

##########################################################################

def fesc(session):
    SESSION_FOLDER_MAP = {
        1: "TIRE", 2: "TIPE", 3: "JARA", 4: "MIKO", 5: "ANRO",
        6: "RIGR", 7: "TUVA", 8: "PEKO", 9: "JAKA"
    }
    FOLDER_NAME = SESSION_FOLDER_MAP.get(session, "ERROR")

    train_json_file = f"{BASE_DIR}/Finnish-emotion-spilits/{FOLDER_NAME}/train.json"
    test_json_file = f"{BASE_DIR}/Finnish-emotion-spilits/{FOLDER_NAME}/test.json"
    dev_json_file = f"{BASE_DIR}/Finnish-emotion-spilits/{FOLDER_NAME}/dev.json"

    with open(train_json_file, "r") as f:
        train_data = json.load(f)
    with open(test_json_file, "r") as f:
        test_data = json.load(f)
    with open(dev_json_file, "r") as f:
        dev_data = json.load(f)

    
    train_df = pd.DataFrame.from_dict(train_data, orient="index").reset_index()
    test_df = pd.DataFrame.from_dict(test_data, orient="index").reset_index()
    dev_df = pd.DataFrame.from_dict(dev_data, orient="index").reset_index()

    train_df["audio"] = train_df["file_path"].apply(update_file_path)
    test_df["audio"] = test_df["file_path"].apply(update_file_path)
    dev_df["audio"] = dev_df["file_path"].apply(update_file_path)

    train_df = train_df.rename(columns={'index': 'file_id', 'label': 'emo'})
    test_df = test_df.rename(columns={'index': 'file_id', 'label': 'emo'})
    dev_df = dev_df.rename(columns={'index': 'file_id', 'label': 'emo'})

    train_df = train_df.loc[train_df['emo'] != '5'].reset_index(drop=True)
    test_df = test_df.loc[test_df['emo'] != '5'].reset_index(drop=True)
    dev_df = dev_df.loc[dev_df['emo'] != '5'].reset_index(drop=True)

    train_df["emo"] = train_df["emo"].apply(update_fesc_label)
    test_df["emo"] = test_df["emo"].apply(update_fesc_label)
    dev_df["emo"] = dev_df["emo"].apply(update_fesc_label)

    labels = ['anger', 'happiness', 'neutral', 'sadness'] # ["1", "2", "3", "4"]
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    train_df["label"] = train_df["emo"].apply(to_label, args=(label2id,))
    test_df["label"] = test_df["emo"].apply(to_label, args=(label2id,))
    dev_df["label"] = dev_df["emo"].apply(to_label, args=(label2id,))

    train_audio_data = Dataset.from_pandas(train_df[['audio', 'label']])
    train_audio_data = train_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    test_audio_data = Dataset.from_pandas(test_df[['audio', 'label']])
    test_audio_data = test_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    dev_audio_data = Dataset.from_pandas(dev_df[['audio', 'label']])
    dev_audio_data = dev_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        'train' : train_audio_data,
        'test' : test_audio_data,
        'dev' : dev_audio_data
    })

    return label2id, id2label, ds

##########################################################################

def cafe(session=None):
    cafe_df = pd.DataFrame()

    # all_files = glob.glob(f"{BASE_DIR}/CaFE_json_splits/*")
    all_files = glob.glob("/content/CaFE/CaFE_splits/*")

    for file in all_files:
        with open(file, "r") as f:
            data = json.load(f)
            
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            df["set"] = [file.split("/")[-1].split(".")[0] for _ in range(len(df))]
            df = df.rename(columns={'wav': 'audio', 'emo': 'label'})
            df["label_flag"] = df["label"].apply(is_common)
            cafe_df = pd.concat([cafe_df, df], axis=0)

    cafe_df = cafe_df.loc[cafe_df["label_flag"] == 1]
    cafe_df = cafe_df.reset_index(drop=True)
    cafe_df["dataset"] = ["CaFE" for _ in range(len(cafe_df))]

    cafe_df_train = cafe_df.loc[cafe_df["set"] != "test"].reset_index(drop=True)
    cafe_df_train = cafe_df_train[["audio", "label"]]

    cafe_df_test = cafe_df.loc[cafe_df["set"] == "test"].reset_index(drop=True)
    cafe_df_test = cafe_df_test[["audio", "label"]]

    cafe_df_train["audio"] = cafe_df_train["audio"].apply(update_cafe_path).values
    cafe_df_test["audio"] = cafe_df_test["audio"].apply(update_cafe_path).values

    train_df = cafe_df_train
    test_df = cafe_df_test

    labels = ['anger', 'happiness', 'neutral', 'sadness']
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # convert categorical label to numerical labels
    train_df["label"] = train_df["label"].apply(to_label, args=(label2id,)).values
    test_df["label"] = test_df["label"].apply(to_label, args=(label2id,)).values

    train_audio_data = Dataset.from_pandas(train_df[['audio', 'label']])
    train_audio_data = train_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    test_audio_data = Dataset.from_pandas(test_df[['audio', 'label']])
    test_audio_data = test_audio_data.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        'train' : train_audio_data,
        'test' : test_audio_data,
        'dev' : test_audio_data
    })

    return label2id, id2label, ds

##########################################################################
