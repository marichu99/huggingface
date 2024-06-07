from transformers import AutoTokenizer, TFAutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
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

# Use HuggingFace embeddings instead of Sentence Transformers
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedder,
)
retriever = vectorstore.as_retriever()

# Step 1: Read code from text file (remains unchanged)
def read_code_from_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

# Step 2: Preprocessing (if necessary) (remains unchanged)
def preprocess_text(text):
    # Perform any necessary preprocessing here (e.g., removing comments, special characters)
    return text

# Step 3: Model and Tokenizer Loading
# Load pre-trained model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Inference using retrieved documents and embeddings
def generate_spring_boot_code(prompt):
    # Use the retriever to fetch relevant documents
    retrieved_docs = retriever.get_relevant_documents(prompt)

    # Concatenate retrieved document contents
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Create the complete input for the model
    complete_input = f"{context}\n\n{prompt}"

    # Chunk the complete input text into smaller chunks
    chunk_size = 512
    input_chunks = [complete_input[i:i+chunk_size] for i in range(0, len(complete_input), chunk_size)]

    # Encode each input chunk separately
    encoded_chunks = [tokenizer.encode(chunk, return_tensors="tf") for chunk in input_chunks]

    # Generate text for each encoded chunk individually
    generated_chunks = []
    for encoded_chunk in encoded_chunks:
        print("The encoded chunk is ",encoded_chunk)
        outputs = model.generate(encoded_chunk, max_length=256, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("The generated text is ",generated_text)
        generated_chunks.append(generated_text)

    # Concatenate the generated text chunks to form the complete output
    generated_code = ''.join(generated_chunks)

    return generated_code

# Main function (mostly unchanged)
def main(file_path, prompt):
    code_text = read_code_from_file(file_path)
    preprocessed_text = preprocess_text(code_text)
    generated_code = generate_spring_boot_code(prompt)
    print(f"Generated Java Spring Boot code: \n{generated_code}")

# Example usage (unchanged)
file_path = "documentserviceImpl.txt"  # Replace with the path to your .txt file
prompt = "Convert the following text into Java Spring Boot code"
main(file_path, prompt)
