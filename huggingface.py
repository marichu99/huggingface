from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

model_local = ChatOllama(model="mistral")

# Load documents from text files
txt_paths = ["documentserviceImpl.txt"]

# Ensure the text file paths are valid and the files exist
valid_txt_paths = [path for path in txt_paths if os.path.isfile(path)]
if len(valid_txt_paths) != len(txt_paths):
    print("Warning: Some text files were not found or invalid.")

# Load valid text files
txt_docs = []
for txt_path in valid_txt_paths:
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            txt_docs.append(Document(page_content=content, metadata={'source': txt_path}))
    except Exception as e:
        print(f"Error loading {txt_path}: {e}")

# Split documents into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(txt_docs)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedder,
)
retriever = vectorstore.as_retriever()

# Step 1: Read code from text file
def read_code_from_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

# Step 2: Preprocessing (if necessary)
def preprocess_text(text):
    # Perform any necessary preprocessing here (e.g., removing comments, special characters)
    return text

# Step 3: Model and Tokenizer Loading
model_name = "bert-base-uncased"  # Example model name, change as needed
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Tokenization
def tokenize_text(text):
    return tokenizer(text, return_tensors="tf")

# Step 5: Inference
def perform_inference(inputs):
    outputs = model(inputs)
    return outputs

# Step 6: Postprocessing (interpret the model's outputs)
def postprocess_outputs(outputs):
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return predicted_class

# Generate Java Spring Boot code from text chunks
def generate_spring_boot_code(prompt):
    # Use the retriever to fetch relevant documents
    retrieved_docs = retriever.retrieve(prompt)

    # Concatenate retrieved document contents
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Create the complete input for the model
    complete_input = f"{context}\n\n{prompt}"

    # Perform inference using the local model
    inputs = tokenizer(complete_input, return_tensors="tf")
    outputs = perform_inference(inputs)
    
    # Process the outputs to get the generated code (assuming model can generate text)
    # Here we assume the model generates text directly. Adjust according to your model's capability.
    generated_code = tokenizer.decode(outputs.logits.numpy(), skip_special_tokens=True)
    return generated_code

# Main function to read file, preprocess, tokenize, and perform inference
def main(file_path, prompt):
    code_text = read_code_from_file(file_path)
    preprocessed_text = preprocess_text(code_text)
    inputs = tokenize_text(preprocessed_text)
    outputs = perform_inference(inputs)
    prediction = postprocess_outputs(outputs)
    print(f"Predicted class: {prediction}")

    # Generate Java Spring Boot code
    generated_code = generate_spring_boot_code(prompt)
    print(f"Generated Java Spring Boot code: \n{generated_code}")

# Example usage
file_path = "documentserviceImpl.txt"  # Replace with the path to your .txt file
prompt = "Convert the following text into Java Spring Boot code"
main(file_path, prompt)
