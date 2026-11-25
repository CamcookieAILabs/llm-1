import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm

# =====================================
# 1. CONFIGURATION
# =====================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100                  # Increased so smaller datasets converge fully
BATCH_SIZE = 16              # Lowered per-step batch to save VRAM for larger blocks
LEARNING_RATE = 0.0006       # Slightly higher peak LR (paired with warmup/decay)
BLOCK_SIZE = 128             # Doubled context window for better coherence

MODEL_DIR = Path("model")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")

print("Training device:", DEVICE)

# =====================================
# 2. LOAD TOKENIZER & INITIALIZE/LOAD MODEL
# =====================================

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab_size = tokenizer.get_vocab_size()
eos_id = tokenizer.token_to_id("<|endoftext|>")

if (MODEL_DIR / "config.json").exists():
    print("Loading existing model weights from model/...")
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
else:
    print("Initializing a fresh small GPT-2 model...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=BLOCK_SIZE,
        n_embd=256,   # Embedding size (compact for fast CPU training)
        n_layer=4,    # 4 transformer blocks (matches ~52 weight tensors)
        n_head=4,     # 4 attention heads
        bos_token_id=eos_id,
        eos_token_id=eos_id
    )
    model = GPT2LMHeadModel(config)

model.to(DEVICE)
model.train()

# =====================================
# 3. LOAD & CHUNK DATABASE
# =====================================

text_files = sorted(Path("training").rglob("*.txt"))

text = "\n\n".join(
    file.read_text(encoding="utf-8")
    for file in text_files
)

encoded = tokenizer.encode(text)
tokens = torch.tensor(encoded.ids, dtype=torch.long)

print("Total tokens:", len(tokens))

if len(tokens) < BLOCK_SIZE:
    raise ValueError("Not enough training data for a single block!")

class TokenDataset(Dataset):
    def __init__(self, token_tensor, block_size):
        num_chunks = len(token_tensor) // block_size
        self.chunks = token_tensor[:num_chunks * block_size].view(num_chunks, block_size)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

dataset = TokenDataset(tokens, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================================
# 4. OPTIMIZER
# =====================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================
# 5. TRAINING LOOP
# =====================================

for epoch in range(EPOCHS):
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in progress_bar:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch,
            labels=batch
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch {epoch + 1}/{EPOCHS} Complete | Average Loss: {average_loss:.4f}")

# =====================================
# 6. SAVE TRAINED MODEL
# =====================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.eval()
model.save_pretrained(str(MODEL_DIR))

print(f"\nTRAINING COMPLETE! Updated weights saved in {MODEL_DIR}/")
