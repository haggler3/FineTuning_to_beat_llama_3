# 🦙 Can Smaller, Efficiently Fine-Tuned LLMs Outperform Larger Models?

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CMUZrz/FineTuning_to_beat_llama_3/blob/main/try_in_colab.ipynb)

**Authors:** Dan Jung, Dhruva Byrapatna, Zachary Zdobinski
_This Repo is a result of the project for the 10-623 Generative AI Course at Carnegie Mellon University._

---

## Highlights & Major Results

This project demonstrates that smaller, efficiently fine-tuned Large Language Models (LLMs) can outperform larger, more computationally expensive counterparts on specialized tasks.

- **Small Model Outperforms Large Model:** Our fine-tuned **Mistral-7B** achieved higher accuracy than the massive **Llama 3 70B** on domain-specific datasets (GSM8K and BeerAdvocate).
- **QLoRA is Highly Effective:** Quantized Low-Rank Adaptation (QLoRA) combined with 4-bit quantization provided significant performance improvements with minimal memory requirements.
- **In-Context Learning (Dr.ICL):** Using Demonstration-Retrieved In-Context Learning (Dr.ICL) alongside QLoRA yielded the best results for text-heavy reasoning tasks (GSM8K & Healthcare).

---

## Quick Start

You can run this project easily locally using **Conda** or in the cloud using **Google Colab**.

### Option A: Google Colab

Click the badge above or open [`try_in_colab.ipynb`](try_in_colab.ipynb) to test the pipeline interactively!

### Option B: Local Conda Environment

We provide a convenient Conda environment.

```bash
# 1. Clone the repository
git clone https://github.com/CMUZrz/FineTuning_to_beat_llama_3.git
cd FineTuning_to_beat_llama_3

# 2. Create the Conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate finetune-llama3
```

_(Alternatively, you can use `pip install -r requirements.txt` if you prefer not to use Conda)._

---

## 🛠 Usage & Pipeline

### 1. Tokenization (`dataset_tokenizer.py`)

Tokenize datasets for standard training or In-Context Learning (ICL) enhanced training.

```bash
python dataset_tokenizer.py --torch_dataset_url "lavita/ChatDoctor-HealthCareMagic-100k" --icl
```

### 2. Fine-Tuning (`Fine_Tune.py`)

Run the QLoRA fine-tuning process. You must have a Hugging Face token exported as `HUGGINGFACE_HUB_TOKEN`.

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
python Fine_Tune.py --project "qlora-run" --user_id "YOUR_HF_USER"
```

### 3. Evaluation (`Evaluate.py`)

Evaluate your model's performance on the test split using BERTScore or other metrics.

```bash
python Evaluate.py --test_dataset "lavita/ChatDoctor-HealthCareMagic-100k" --use_icl
```

---

## 📊 Experimental Results

| Model                 | Dataset    | F1 / Acc      |
| :-------------------- | :--------- | :------------ |
| **Mistral Base**      | Healthcare | 0.83 (F1)     |
| **Mistral QLoRA**     | Healthcare | 0.87 (F1)     |
| **Mistral ICL+QLoRA** | Healthcare | **0.89 (F1)** |
| **Llama 3 70B**       | Healthcare | 0.87 (F1)     |

_(See full table in previous versions or paper for other datasets like Beer and Math)._

---

## Method Overview

1. **Parameter-Efficient Fine-Tuning (QLoRA):** 4-bit NormalFloat Quantization + Paged Optimizers + LoRA. Adapts the smaller model efficiently.
2. **Demonstration-Retrieval for In-Context Learning (Dr.ICL):** Retrieves semantically similar examples and prepends them to the prompt to dynamically improve performance.

---

## Future Work
- **Refine Data Quality:** Improve prompt design to further enhance combined QLoRA/Dr.ICL.
- **New Domains:** Evaluate on legal, medical, or other highly technical contexts.

---

## Key Related Work

Our work is built upon the insights from several foundational papers in the field of efficient model training and in-context learning:

* **AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning** (Cui et al., 2024): Introduced a methodology for optimizing LLMs for domain-specific tasks by creating precise, tailored datasets.
* **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023): The core technical paper for the QLoRA method, which formed the backbone of our fine-tuning approach.
* **Dr.ICL: Demonstration-Retrieved In-Context Learning** (Luo et al., 2023): Proposed the method of retrieving demonstrations to boost in-context learning, which we implemented and tested.
* **Textbooks Are All You Need** (Gunasekar et al., 2023): Showcased the power of training on high-quality, "textbook" style data to achieve strong performance with smaller models.
* **Large Dual Encoders Are Generalizable Retrievers** (Ni et al., 2021): This work supported our choice of the GTR-T5 model as a dense retriever for our Dr.ICL implementation, showing its effectiveness in zero-shot retrieval.
