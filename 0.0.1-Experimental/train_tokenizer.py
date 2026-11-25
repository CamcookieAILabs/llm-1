from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# =====================================
# 1. CONFIGURATION
# =====================================

VOCAB_SIZE = 16000  # Adjust based on dataset size (e.g., 8000 to 32000)
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]
OUTPUT_DIR = Path("tokenizer")

# =====================================
# 2. INITIALIZE BYTE-LEVEL BPE TOKENIZER
# =====================================

tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True
)

# =====================================
# 3. GATHER FILES & TRAIN
# =====================================

text_files = [str(f) for f in sorted(Path("training").rglob("*.txt"))]

if not text_files:
    raise FileNotFoundError("No .txt files found in the 'training' folder!")

print(f"Training tokenizer on {len(text_files)} files...")
tokenizer.train(files=text_files, trainer=trainer)

# =====================================
# 4. SAVE TOKENIZER
# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tokenizer_path = OUTPUT_DIR / "tokenizer.json"
tokenizer.save(str(tokenizer_path))

print(f"\nTOKENIZER COMPLETE! Saved to {tokenizer_path}")
