from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
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

# 1. Split data into chunks

# URLs to load
urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]

# Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

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

# Combine web and text file documents
all_docs_list = docs_list + txt_docs

# Split documents into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(txt_docs)

# 2. Convert documents to Embeddings and store them
openai_api_key = ""  # Replace this with your actual OpenAI API key
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedder,
)
retriever = vectorstore.as_retriever()

# 3. Before RAG
print("Before RAG\n")
before_rag_template = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))

# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": Runnable()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("Change the codebase into spring boot"))
