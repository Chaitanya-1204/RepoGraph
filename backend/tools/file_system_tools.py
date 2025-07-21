import os 
import git 


CLONED_REPOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cloned_repos')

def clone_repo(repo_url ):
    
    """
    Clones a public Git repository to a local directory.

    If the repository has already been cloned, it returns the path to the existing
    directory without re-cloning.

    Args:
        repo_url: The URL of the public Git repository to clone.

    Returns:
        The local path to the cloned repository, or None if cloning fails.
    """
    
    
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    local_path = os.path.join(CLONED_REPOS_DIR, repo_name)  
    
    
    # check if repo already exist or not
    
    if os.path.exists(local_path):
        print("Repo Already Exist in local")
        return local_path
    
    # Clone the repo to local_path 
    print(f"Cloning Repo : {repo_url}")
    git.Repo.clone_from(repo_url , local_path )
    
    return local_path


def list_files_in_directory(directory_path):
    
    """
    Lists all files and subdirectories in a given directory.

    Args:
        directory_path: The absolute or relative path to the directory.

    Returns:
        A list of names of files and directories, or None if the path is invalid.
    """
    
    return os.listdir(directory_path)


def read_file_content(file_path):
    
    with open(file_path , "r"  , encoding="utf-8") as f :
        return f.read()
    
    
    
    

    
repo_url = "https://github.com/langchain-ai/langchain"
