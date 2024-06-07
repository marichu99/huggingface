from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Read input data from files
file_paths = ["documentserviceImpl.txt"]
input_texts = []
for file_path in file_paths:
    with open(file_path, "r") as file:
        input_texts.append(file.read())

# Tokenize input data
input_token_ids = [tokenizer.encode(text, return_tensors="pt") for text in input_texts]

# Model inference
output_texts = []
for input_token_id in input_token_ids:
    output = model.generate(input_token_id, max_new_tokens=100, num_return_sequences=1, do_sample=True, temperature=0.7)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    output_texts.append(output_text)

# Store or handle the model output
for output_text in output_texts:
    print(output_text)
