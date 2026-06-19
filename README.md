<h1 align="center">Multi-Teacher Language-Aware Knowledge Distillation for Speech Emotion Recognition</h1>

<p align="center">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-blue" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/%F0%9F%94%A5-PyTorch-purple" alt="PyTorch" />
</p>

<div align="center">
  <details>
    <summary>Table of Contents</summary>
    <a href="#run-on-colab">Run on Colab</a><br>
    <a href="#installation">Installation</a><br>
    <a href="#usage">Usage</a><br>
    <a href="#contributing">Contributing</a><br>
    <a href="#citation">Citation</a>
  </details>
</div>

## Run on Colab
* Run the notebook: [**mtkd4ser.ipynb**](https://github.com/aalto-speech/mtkd4ser/blob/EN_FI_FR/mtkd4ser.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aalto-speech/mtkd4ser/blob/EN_FI_FR/mtkd4ser.ipynb)



## Installation

**1. Clone the Repository**
```bash
git clone https://github.com/aalto-speech/mtkd4ser.git
```
```bash
cd mtkd4ser
```

**2. Create the Environment**
```bash
conda env create -f environment.yml
```

**3. Activate the Environment**
```bash
conda activate ser_venv
```

<!--
**x. xx**
```bash

```
-->

## Usage

**1. Multi-Teacher Language-Aware Knowledge Distillation for English Speech Emotion Recognition Using the Monolingual Setup**
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 5 --TRAINING 1 --PARADIGM "MTKD" --LANGUAGE "EN" --LINGUALITY "Monolingual"
```

**2. Conventional Knowledge Distillation for Finnish Speech Emotion Recognition Using the Multilingual Setup**
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 9 --TRAINING 1 --PARADIGM "KD" --LANGUAGE "FI" --LINGUALITY "Multilingual"
```

**3. Vanilla Fine-Tuning for French Speech Emotion Recognition Using the Multilingual Setup**
```bash
python main.py --LEARNING_RATE 3e-5 --BATCH_SIZE 16 --N_EPOCHS 20 --SESSION 1 --TRAINING 1 --PARADIGM "FT" --LANGUAGE "FR" --LINGUALITY "Multilingual"
```

**4. Available Configurations and Choices**

It supports a range of configurable parameters for training, validation, and evaluation. The table below details each *Configuration* and its *options*. Select the options that fit your use case.
| **Configuration** | **Options**                       |
|:------------------|:----------------------------------|
| LINGUALITY        | `Monolingual` or `Multilingual`   |
| LANGUAGE          | `EN` or `FI` or `FR`              |
| PARADIGM          | `MTKD` or `KD` or `FT`            |
| TRAINING          | `1` or `0`                        |
| SESSION           | `EN: 1-5` or `FI: 1-9` or `FR: 1` |
| N_EPOCHS          | `ℤ⁺`                              |
| BATCH_SIZE        | `ℤ⁺`                              |
| LEARNING_RATE     | `ℝ⁺`                              |


## Contributing
- [x] MTKD-based monolingual SER methods for English, Finnish, and French.
  - [ ] Adapt the method for a new language (e.g., Chinese).
- [x] MTKD-based multilingual SER method for English, Finnish, and French.
  - [ ] Extend the multilingual method to include a resource-scarce language (e.g., Bangla).   
- [ ] Incorporate heterogeneous Large Audio-Language Models in the MTKD method.
  - [ ] Distill the internal knowledge of heterogeneous models to the student.


## Citation



