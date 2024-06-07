from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate text based on a prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

# Decode and print generated text
for i, sample_output in enumerate(output):
    print(f"Generated Text {i+1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
