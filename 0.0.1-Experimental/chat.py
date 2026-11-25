
import torch
from transformers import GPT2LMHeadModel
from tokenizers import Tokenizer

# =================================
# 1. LOAD YOUR TOKENIZER
# =================================

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

# =================================
# 2. LOAD YOUR TRAINED MODEL
# =================================

device = torch.device("cpu")

model = GPT2LMHeadModel.from_pretrained("model")
model.to(device)
model.eval()

print("Your AI is ready!")
print("Type 'exit' to quit.")

# =================================
# 3. CHAT LOOP
# =================================

while True:

    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        break

    # Convert text into token IDs
    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long
    ).to(device)

    if input_ids.shape[1] == 0:
        print("Please enter some text.")
        continue

    # Generate a response
    with torch.no_grad():

        output = model.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            pad_token_id=tokenizer.token_to_id("[PAD]")
        )

    # Only decode newly generated tokens
    generated_tokens = output[0][input_ids.shape[1]:]

    response = tokenizer.decode(
        generated_tokens.tolist()
    )

    print("\nAI:", response)
