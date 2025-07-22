import os
import json

# Vector Store

from langchain_community.vectorstores import Chroma

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Text Splitter and Documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.text_splitter import Language


# Allowed extensionsced 

ALLOWED_EXTENSIONS = {
    ".c", ".cpp", ".java", ".py",
    ".js",
    ".html", ".xml",
    ".css",
    ".ipynb",
    ".md", ".txt",
    ".yml", ".yaml"
}


#  Chunking parameters

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


# Language Specific Splitters

# Python
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# javascript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# C and CPP
c_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.C,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# JAVA 
java_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA,
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
)

# Recursive Splitters


# Html
html_splitter = RecursiveCharacterTextSplitter(
    separators=["</section>", "</article>",
                "</div>", "</p>", "<br>", "\n\n", "\n", " "],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# Markdown
md_splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n### ", "\n#### ", "\n* ", "\n\n", "\n", " "],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# CSS
css_splitter = RecursiveCharacterTextSplitter(
    separators=["\n}", "\n\n", "\n", " "],  # Split on closing brace of a rule
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Generic Splitter for yaml and txt files 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
)


# Markdown Splitters 
def chunk_notebook(notebook_content: str):
    """
    Parses a .ipynb file and chunks its code and markdown cells separately.
    """
    notebook = json.loads(notebook_content)
    chunks = []
    
    for i, cell in enumerate(notebook.get('cells', [])):
        
        cell_source = "".join(cell.get('source', []))
        
        metadata = {"cell_number": i + 1}
        if cell['cell_type'] == 'code':
            # Use the Python splitter for code cells
            chunks.extend(py_splitter.create_documents([cell_source] , metadatas=[metadata]))
            
            
        elif cell['cell_type'] == 'markdown':
            # Use the Markdown splitter for text cells
            chunks.extend(md_splitter.create_documents([cell_source], metadatas=[metadata]))
            
            
    return chunks



# map the extension with chunkiung map 


CHUNKER_MAP = {
    ".py": py_splitter,
    ".js": js_splitter,
    ".c": c_splitter,
    ".cpp": c_splitter, 
    ".java": java_splitter,
    ".html": html_splitter,
    ".xml": html_splitter,  
    ".css": css_splitter,
    ".md": md_splitter,
    ".txt": text_splitter,
    ".yml": text_splitter,  
    ".yaml": text_splitter,
    ".ipynb": chunk_notebook,  
}




def create_vector_store(repo_path):
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
    
    repo_name = os.path.basename(repo_path)
    persist_directory = f"vector_stores/{repo_name}"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store for '{repo_name}' from '{persist_directory}'")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
        return vector_store
    
    

    # Getting all the files and converting them into langchain docments
    documents = []

    # traversing through the repo
    for root, _, files in os.walk(repo_path):

        if ".git" in root:
            continue

        for file in files:
            # For each file creating a langchain document and adding metadata which is its source path
            file_path = os.path.join(root, file)
            # Corrected typo from splittext to splitext
            file_extension = os.path.splitext(file)[1]

            if file_extension in ALLOWED_EXTENSIONS:

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    doc = Document(page_content=content,
                                   metadata={
                                       "source": file_path , 
                                        "extension" : file_extension , 
                                        "repo_name": repo_name
                                    }
                                )

                    documents.append(doc)
                except Exception:
                    # Ignore files that cannot be read
                    continue
    
    print(f"Found {len(documents)} documents to process.")
    
    # Spliting or Chunking all the documents using Specific Text Splitter
    all_chunks = []
    for doc in documents:

        # Gwet the extension 
        file_extension = doc.metadata["extension"]
        
        # get the chunker for the file extension 
        chunker = CHUNKER_MAP.get(file_extension)
        
        # For .ipynb file
        if callable(chunker):
            
            notebook_chunks = chunker(doc.page_content)
            for chunk in notebook_chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(notebook_chunks)
        
        # for others  
        else:
            
            chunks = chunker.split_documents([doc])
            all_chunks.extend(chunks)
            
    print(f"\nAdding {len(all_chunks)} document chunks to the vector store")

    # build the vector store
    vector_store = Chroma(

        embedding_function=embeddings,
        persist_directory=persist_directory
    )
       
    batch_size = 100
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size 

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size] 
        vector_store.add_documents(batch)
    
    print("Successfully added all documents to ChromaDB.")

    

    return vector_store


def search_codebase(query, vector_store):
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
