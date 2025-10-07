from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

os.makedirs("chroma_db", exist_ok=True)

# change this to PyPDFLoader(...) if you have PDFs in data/
loader = TextLoader("data/notes.txt", encoding="utf8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# local/free embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("Ingested", len(chunks), "chunks.")

