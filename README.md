<p align="center">
  <img src="logo.svg" alt="Camcookie AI Labs logo" width="120">
</p>

# Camcookie AI Labs LLM

**Version:** 0.0.1-Experimental

**Status:** Experimental


A small, locally run language model for experimenting with next-token text generation. This is an early prototype, not a production assistant; its quality has not been formally evaluated.

## At a glance

LLM 0.0.1 is a decoder-only GPT-2 language model. It reads a sequence of tokens and predicts likely next tokens, one at a time. Text is encoded with a byte-level BPE tokenizer.

| Model detail | Configuration |
| --- | --- |
| Architecture | GPT-2 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Embedding width | 256 |
| Maximum context | 128 tokens |
| Vocabulary | 16,000 tokens |
| Weight format | Float32 |
| Training objective | Causal next-token prediction |

## Training data

The checked-in corpus contains public-domain Jane Austen books and conversational text. Both training scripts recursively read `.txt` files under `training/`; text stored elsewhere is not included automatically.

## Setup and training

> **We could not upload the training data we used.** The training dataset was to large to be uploaded to github. Because of that, we are providing a ZIP version of the model for if you want to use the dataset we used. You can find it in the relese notes.

Run commands from this version's directory. Create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current tokenizer is already available at `tokenizer/tokenizer.json`.

Training tokenizes the corpus and splits it into consecutive 128-token blocks. The training configuration is:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `0.0006` |
| Epochs | 100 |
| Batch size | 16 |
| Device | CUDA when available, otherwise CPU |

Start training with the existing tokenizer:

```powershell
python train_model.py
```

If `model/config.json` exists, the training script loads the existing model weights and continues training. Otherwise, it initializes a new model. The updated model is saved to `model/` when training completes; back up that directory before training if you want to preserve the current weights.

### Rebuilding the tokenizer

To train a tokenizer from the current `training/` text files:

```powershell
python train_tokenizer.py
```

The tokenizer trainer targets a vocabulary of 16,000 tokens. Rebuilding it can change token IDs, so the existing model weights may no longer match. The model must be initialized or otherwise made compatible with the rebuilt tokenizer before training or inference; do not pair a changed tokenizer with old weights.

## Chat

Start the interactive chat from this directory:

```powershell
python chat.py
```

The chat script runs on CPU and samples up to 40 new tokens per prompt, using temperature `0.8` and top-k `40`. Type `exit` to quit. Each prompt is handled independently; previous turns are not included, so the model has no conversation memory.

### Example

One real response from the current model:

```text
You: Hello!
AI: it would not be a bad sore throat, but he would be a favourite with her. if she had done it again, he only came out to her own room, he would be at home,
```

This illustrates the model's experimental quality; responses can be incoherent or unrelated to the prompt.

## Limitations

This version is suitable for local experiments with small language models and training workflows. It may produce incoherent, repetitive, inaccurate, or unexpected text. Its 128-token context is short, and the training data does not guarantee factual knowledge or reliable instruction following. Do not rely on its output for important decisions.

## License

© 2026 Camcookie AI Labs. Licensed under the [MIT License](LICENSE).

The training corpus includes public-domain works and conversational text; third-party material remains subject to its own rights and notices.