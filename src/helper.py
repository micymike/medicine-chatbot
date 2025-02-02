from langchain_core.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from unittest import loader


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Extract pdf data



def load_pdf(data):
    directory_loader = DirectoryLoader(data, 
                                       glob="*.pdf",
                                       loader_cls=PyPDFLoader)
    
    documents = directory_loader.load()

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings



    