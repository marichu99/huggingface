from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os

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

# Load documents from PDFs
pdf_paths = ["document.txt", "documentController.txt", "documentservice.txt", "documentserviceImpl.txt"]
# Ensure the PDF paths are valid and the files exist
valid_pdf_paths = [path for path in pdf_paths if os.path.isfile(path)]
if len(valid_pdf_paths) != len(pdf_paths):
    print("Warning: Some PDF files were not found or invalid.")
    
# Load valid PDFs
pdf_docs = []
for pdf_path in valid_pdf_paths:
    try:
        pdf_docs.extend(PyPDFLoader(pdf_path).load())
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")
# pdf_docs = [PyPDFLoader(pdf_path).load() for pdf_path in pdf_paths]
pdf_docs_list = [item for sublist in pdf_docs for item in sublist]

# Combine web and PDF documents
all_docs_list = docs_list + pdf_docs_list

# Split documents into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(pdf_docs_list)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedder
)
retriever = vectorstore.as_retriever()

# 3. Before RAG
print("Before RAG\n")
before_rag_template = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Change the codebase into spring boot"}))

# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("Change the codebase into spring boot"))
