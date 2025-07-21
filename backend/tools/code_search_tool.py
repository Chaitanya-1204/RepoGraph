import os 

# Vector Store
import chromadb
from langchain_community.vectorstores import Chroma

# Embeddings 
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Text Splitter and Documents 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.text_splitter import Language

ALLOWED_EXTENSIONS = [
    '.py', '.js', '.java', '.c', '.cpp', '.go', '.rb', '.php', '.ts', '.tsx', '.jsx', '.html', '.css', '.md',  # langguage Specific text Spliter 
    '.json', '.yml', '.yaml', '.sh' # normal text splitter
]

# Map file extensions to the Language for the splitter
LANGUAGE_MAP = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.java': Language.JAVA,
    '.c': Language.C,
    '.cpp': Language.CPP,
    '.go': Language.GO,
    '.rb': Language.RUBY,
    '.php': Language.PHP,
    '.ts': Language.TS,
    '.tsx': Language.TSX,
    '.jsx': Language.JS,
    '.html': Language.HTML,
    '.css': Language.CSS,
    '.md': Language.MARKDOWN,
}


def code_rag(repo_path):
    """
        The entire process of creating a code-aware RAG pipeline.
            1. Discovers and loads code files.
            2. Splits them into meaningful chunks based on language.
            3. Creates vector embeddings and stores them in a persistent ChromaDB.
        
        Args:
            repo_path: The local file path to the cloned repository.

        Returns:
            A Chroma vector store object ready for querying.
    """
    
    
    # Getting all the files and converting them into langchian docments
    documents = []
    
    # traversing through the repo 
    for root , _ , files in os.walk(repo_path):
        
        if ".git" in root: 
            continue
        
        
    
        for file in files:
            # For each file creating a langchain document and adding metadata which is its source path 
            file_path = os.path.join(root , file)
            file_extension = os.path.splittext(file)[1]
            
            if file_extension in ALLOWED_EXTENSIONS:
                
                with open(file_path , "r" , encoding = "utf-8") as f:
                    content = f.read()
                
                
                doc  = Document(page_content = content , 
                                metadata = {"source" : file_path})
                
                documents.append(doc)
    
    # Splitting or Chunking all the documents using language specific Text Splitter    
    all_chunks = []
    for doc in documents:
    
        file_extension = os.path.splittext(doc.metadata["source"])[1]
        # get the language from the language map 
        language = LANGUAGE_MAP.get(file_extension)
        
        # Split the doc based on Language Specific Text Splitter
        if language:
            
            splitter = RecursiveCharacterTextSplitter.from_language(
                language = language,
                chunk_size = 2000 , 
                chunk_overlap = 200
            )
        # Split the doc based on normal Text splitter 
        else:
            splitter = RecursiveCharacterTextSplitter(
                
                chunk_size = 2000 , 
                chunk_overlap = 200
            )
            
        # Finally chunk the doc 
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        
        
    
    # Vector Store Creation 
    
    repo_name = os.path.basename(repo_path)
    
    persist_directory = f"vector_stores/{repo_name}"
    
    # Getting the embedding model from hugging face 
    
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # build the vector store 
    vector_store = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store


    
    

def search_codebase(query , vector_store) :
    
    """
    Performs a semantic search on the provided Chroma vector store.
    This is the function that the LangChain agent will actually call.

    Args:
        query: The user's natural language question.
        vector_store: The Chroma vector store object for the repository.

    Returns:
        A formatted string of the most relevant code chunks.
    """
    
    
    # Similarity search 
    results = vector_store.similarity_search(query, k=5)
    
    # building a single context string for LLM 
    context_string = ""
    for i, doc in enumerate(results):
        context_string += f"--- Result {i+1} ---\n"
        context_string += f"Source File: {doc.metadata.get('source', 'Unknown')}\n\n"
        context_string += f"Content:\n{doc.page_content}\n\n"
        
    return context_string