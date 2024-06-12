from transformers import AutoTokenizer, TFAutoModelForCausalLM, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Function to read code from a file
def read_code_from_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

# Function to preprocess text if needed
def preprocess_text(text):
    # Perform any necessary preprocessing here (e.g., removing comments, special characters)
    return text

# Function to generate rewritten code
def rewrite_code(prompt, original_code):
    # Retrieve the API token from environment variable
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if api_token is None:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set in the environment variables")

    # Authenticate using the token
    login(token=api_token)

    "huggingface/models/"
    # Model and Tokenizer Loading
    model_name = "meta-llama/Meta-Llama-3-70B"
    model = AutoModelForCausalLM.from_pretrained(model_name) 
    code_generation_model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the complete input for the model
    complete_input = f"{prompt}\n\n{original_code}"

    # Encode the input text
    encoded_input = tokenizer.encode(complete_input, return_tensors="tf")

    # Generate rewritten code
    outputs = model.generate(encoded_input, max_length=512, num_return_sequences=1)
    rewritten_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return rewritten_code

# Main function
def main(file_path, prompt):
    code_text = read_code_from_file(file_path)
    preprocessed_text = preprocess_text(code_text)
    rewritten_code = rewrite_code(prompt, preprocessed_text)
    print(f"Rewritten code: \n{rewritten_code}")

# Example usage
file_path = "documentserviceImpl.txt"  # Replace with the path to your .txt file containing the Java EE code
prompt = "Convert the following Java EE code to Spring Boot code"  # Customize your rewriting prompt
main(file_path, prompt)
