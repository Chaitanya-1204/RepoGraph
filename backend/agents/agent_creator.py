import os
import sys
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from dotenv import load_dotenv
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_system_tools import clone_repo, list_files_in_directory, read_file_content
from tools.code_search_tool import search_codebase
from tools.visualizer_tool import generate_graph

load_dotenv()

def create_agent(vector_store):
    # Explicitly get the API key from the environment
    
    
    search_tool_func = partial(search_codebase, vector_store=vector_store)
    tools = [
        Tool(
            name="Codebase Search",
            func=search_tool_func,
            description=(
                """ 
                Use this tool to semantically search across the entire cloned codebase.
                This is especially useful when the relevant implementation details, functions, classes, or comments may be spread across multiple files, or when the exact file or symbol name is unknown.
                
                Input should be the user's original question or a refined natural language query. 
                Returns the most relevant code chunks based on semantic similarity
                """
                
            )
        ),
        
        Tool(
            name="Clone_Repository",
            func=clone_repo,
            description="""
                Use this tool to clone a public GitHub repository to the local filesystem.
                Input: A valid GitHub repository URL (e.g., "https://github.com/user/repo.git" or without ".git").
                Behavior:
                - If the repository is already cloned, it returns the existing local path without cloning again.
                - If the repository is not yet cloned, it clones it and returns the full local path.
                Call this tool first before attempting to read or list any files from a remote repository.
                """
        ),
        Tool(
            name="List_Files",
            func=list_files_in_directory,
            description="""
                Use this tool to list all files and subdirectories within a given directory path.
                Input: An absolute or relative path to a local directory (e.g., "./cloned_repos/repo_name/").
                Output: A list of file and folder names inside the directory. Returns None if the path is invalid.
                This is useful for exploring the structure of a cloned repository.
            
            """
        ),
        Tool(
            name="Read_File",
            func=read_file_content,
            description="""
                Use this tool to read the complete content of a file as plain text.
                Input: The full path to a valid, readable text file (e.g., "./cloned_repos/repo_name/app.py").
                Output: The file’s entire content as a string. Useful for analyzing code, configuration, or documentation files.
                Only use this after identifying the file path using the List_Files tool.
            """
        ),
        # Tool(
        #     name="Generate_Graph",
        #     func=generate_graph,
        #     description="""
        #         Use this tool to generate a visual diagram that represents relationships between files, classes, or functions.
        #         Input: A valid Mermaid.js diagram string (e.g., flowchart, sequence, or class diagram syntax).
        #         Behavior:
        #         - It does not analyze or validate the graph.
        #         - It simply passes the diagram syntax to the frontend for rendering.
        #         Use this tool as the final step when you want the user to see a visual representation of the system structure based on your analysis.
            
        #         """
        # )
    ]

    # # A more robust prompt to force the ReAct format.
    # prompt_template = """
    # You are an expert software engineer AI assistant. Your goal is to answer a user's question about a codebase by following a strict reasoning process.

    # ---
    # TOOLS:
    # You have access to the following tools:
    # {tools}

    # ---
    # RESPONSE FORMAT INSTRUCTIONS:
    # To use a tool, you MUST use the following format, with no conversational text before or after.

    # ```
    # Thought: Do I need to use a tool? Yes. I need to [reason for using the tool].
    # Action: The action to take, should be one of [{tool_names}]
    # Action Input: The input to the action
    # Observation: The result of the action
    # ```

    # When you have the final answer and do not need to use a tool anymore, you MUST use this exact format:

    # ```
    # Thought: Do I need to use a tool? No. I have all the information required.
    # Final Answer: [Your final answer here. It MUST include both a text explanation and a Mermaid.js graph code block.]
    # ```
    # ---

    # Begin!

    # Question: {input}
    # Thought: {agent_scratchpad}
    # """
    # prompt_template = """
    #         You are an expert AI software engineer assistant tasked with answering questions about a codebase.
    #         You must use a structured ReAct format to reason step by step, utilize tools when needed, and provide clear, insightful answers.

    #         ---
    #         TOOLS:
    #         You can use the following tools to help answer questions:
    #         {tools}

    #         These tools let you search and retrieve parts of the codebase, documentation, or perform custom logic.

    #         ---
    #         RESPONSE FORMAT INSTRUCTIONS:
    #         Whenever answering a question, follow this structured reasoning loop. **Do not break format**.

    #         If you need to use a tool, use this format:
            
    #         Thought: Do I need to use a tool? Yes. I need to [reason for tool use].
    #         Action: [Tool name here, must be one of {tool_names}]
    #         Action Input: [Input to the tool, e.g. “Where is the image captioning model defined?”]
    #         Observation: [Result returned by the tool]

    #         You can repeat the tool usage loop multiple times.

    #         When you have gathered everything needed and are ready to answer, use this final format:
            
            
    #         Thought: Do I need to use a tool? No. I have all the information required.
    #         Final Answer:
    #         [Final answer must include:]

    #         • A concise explanation of the answer.
            
            
    #         ---
    #         EXAMPLES OF QUESTIONS YOU MAY BE ASKED:

    #         These are sample questions you may receive from users. Your reasoning loop should apply effectively to any of them:

    #         1. What is the purpose of the BLIP model and how is it architected?
    #         2. Where is the image captioning logic defined and how does it work?
    #         3. Which file contains the vision encoder used by BLIP?
    #         4. Where is the contrastive loss function implemented?
    #         5. How is the tokenizer initialized in the BLIP model?
    #         6. Which training script is used for visual question answering?
    #         7. Where can I find the caption generation function in the code?
    #         8. Which datasets are used for training and how are they loaded?
    #         9. How does BLIP handle inference for image-text retrieval?
    #         10. What are the available command-line arguments for training?
    #         11. What is the relationship between BLIP and BLIP-2 in this repo?
    #         12. Where is the model checkpoint loaded and which class uses it?
    #         13. How is cross-modal attention implemented in BLIP?
    #         14. How does the BLIP model handle pretraining with noisy web data?
    #         15. Where is the `generate` function used and how can I modify it?

    #         ---
    #         Begin!

    #         Question: {input}
    #         {agent_scratchpad}

            

    # """

    prompt_template = """  
        You are an expert AI software engineer assistant tasked with answering questions about a codebase.
        You must use a structured ReAct format to reason step by step, utilize tools when needed, and provide clear, insightful answers.

        ---
        TOOLS:
        You have access to the following tools to investigate the codebase:
        {tools}

        ---
        RESPONSE FORMAT INSTRUCTIONS:
        Always follow this structured reasoning loop. **Do not break format**.

        **1. Thought:** First, think about the user's question and devise a plan. Do you need to use a tool? If so, what is the goal?

        **2. Action (Optional):** If you need to use a tool, use this block.

        * **Action:** [Tool name, must be one of {tool_names}]
        * **Action Input:** [Input to the tool]

        **3. Observation:** This block will contain the result returned by the tool.

        You can repeat the **Thought -> Action -> Observation** loop as many times as necessary to gather all the information required to answer the user's question.

        When you have gathered enough information and are ready to answer, you must use the final block.

        ---

        **4. Final Answer:**
            Your final answer must be a clear, well-structured explanation in simple English. Follow this format:

            * **Summary:** [Provide a one or two-sentence summary of the answer.]

            * **Key Findings:**
            * **[Finding 1]:** [Describe the first key point.]
            * **[Finding 2]:** [Describe the second key point.]
            * **...**

            * **Sources:**
            * `[path/to/source/file1.py]`
            * `[path/to/source/file2.py]`
            
            * **[If you could not find the answer after using the tools, clearly state that the information was not found in the provided codebase.]**

        ---
        EXAMPLES OF QUESTIONS YOU MAY BE ASKED:

        1.  What is the purpose of the BLIP model and how is it architected?
        2.  Where is the image captioning logic defined and how does it work?
        3.  Which file contains the vision encoder used by BLIP?
        4.  Where is the contrastive loss function implemented?
        5.  How is the tokenizer initialized in the BLIP model?
        6.  Which training script is used for visual question answering?
        7.  Where can I find the caption generation function in the code?
        8.  Which datasets are used for training and how are they loaded?
        9.  How does BLIP handle inference for image-text retrieval?
        10. What are the available command-line arguments for training?
        11. What is the relationship between BLIP and BLIP-2 in this repo?
        12. Where is the model checkpoint loaded and which class uses it?
        13. How is cross-modal attention implemented in BLIP?
        14. How does the BLIP model handle pretraining with noisy web data?
        15. Where is the `generate` function used and how can I modify it?
        
        ---
        EXAMPLE REASONING LOOPS:
        
        Here are examples of how you should reason and respond to different types of questions.

        **Example 1: Finding a specific piece of code.**

        Question: Which file contains the vision encoder used by BLIP?

        Thought: I need to find the file that defines the "vision encoder" for the BLIP model. The best first step is to use the semantic search tool to look for relevant code chunks across the entire codebase.
        Action: Codebase Search
        Action Input: "BLIP model vision encoder"
        Observation: 
        --- Result 1 ---
        Source File: ./models/blip.py
        Content:
        class BlipVisionModel(nn.Module):
            def __init__(self, config: BlipVisionConfig):
                super().__init__()
                self.vision_model = VisionTransformer(...)
        ---

        Thought: The search result strongly indicates that the vision encoder is defined in the file `./models/blip.py` within the `BlipVisionModel` class. I have enough information to answer the user's question directly.
        
        Final Answer:
            **Summary:** The vision encoder for the BLIP model is implemented in the `blip.py` file.

            **Key Findings:**
            * The core logic is located within the `BlipVisionModel` class.
            * This class initializes a `VisionTransformer` which serves as the main visual feature extractor.

            **Sources:**
            * `./models/blip.py`
            
        
            
        **Example 2: Understanding the repository structure.**

        Question: What is the high-level directory structure of this repository?

        Thought: The user wants to understand the layout of the repository. The `List_Files` tool is perfect for this. I will use it on the root directory to see the top-level files and folders.
        Action: List_Files
        Action Input: .
        Observation: ['models', 'scripts', 'configs', 'README.md', 'requirements.txt']

        Thought: I have the top-level directory structure. This is enough to give a high-level overview. I will list these and briefly explain what each likely contains based on common software engineering conventions.
        
        
        Final Answer:
            **Summary:** The repository is organized into standard directories for models, scripts, and configurations.

            **Key Findings:**
            * `models/`: This directory likely contains the core model definitions, such as the BLIP architecture.
            * `scripts/`: This probably holds utility scripts for training, evaluation, or data processing.
            * `configs/`: This directory likely contains configuration files for different experiments or model setups.
            * `README.md`: This is the main documentation file for the project.
            * `requirements.txt`: This file lists the necessary Python packages to run the code.

            **Sources:**
            * N/A (Information gathered from directory listing)
            
            
            
        
        **Example 3: Answering a complex question requiring multiple steps and tools.**

        Question: How does the training script use the configuration file to build the BLIP model?

        Thought: This is a multi-step question. First, I need to find the main training script. Then, I need to see how it reads a configuration. Finally, I need to find where it uses that configuration to instantiate the model. I'll start by searching for "training script".
        Action: Codebase Search
        Action Input: "training script"
        Observation: 
        --- Result 1 ---
        Source File: ./scripts/train.py
        Content:
        def main():
            config = parse_config()
            model = create_model(config)
            ...
        ---

        Thought: The observation shows that `scripts/train.py` is the training script. It calls `parse_config()` and then `create_model()`. Now I need to investigate the `create_model` function to see how the model is built. I will read the full content of `scripts/train.py`.
        Action: Read_File
        Action Input: ./scripts/train.py
        Observation: [Content of train.py is returned, including the following]
        from models.blip import BlipForQuestionAnswering

        def create_model(config):
            model = BlipForQuestionAnswering(config=config.model)
            return model

        Thought: I have all the pieces now. `scripts/train.py` reads a config, which is then passed to the `BlipForQuestionAnswering` class from `models/blip.py` to create the model instance. I have enough information to construct the final answer.
        Final Answer:
        **Summary:** The training script uses a configuration file to instantiate the `BlipForQuestionAnswering` model class.

        **Key Findings:**
        * The main training process is initiated in `scripts/train.py`.
        * This script first parses a configuration file to get model parameters.
        * The configuration object is then passed directly to the `BlipForQuestionAnswering` class, which is imported from `models/blip.py`.
        * This class acts as a factory to construct the final BLIP model instance, ready for training.

        **Sources:**
        * `./scripts/train.py`
        * `./models/blip.py`

        ---

        ---
        Begin!

        Question: {input}
        {agent_scratchpad}
    
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add it to proceed.")

    # Pass the API key directly to the client for robust handling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor



# def preprocess(repo_url):
#     pass
    
    
# if __name__ == '__main__':
#     agent_executor = create_agent()

#     repo_url = "https://github.com/salesforce/BLIP.git" # Using a smaller repo for faster testing
#     test_query = f"Clone {repo_url}, list the files in its root directory, and then create a simple graph showing the repo and some of its root files."

#     print(f"--- Running Test Query ---")
#     print(f"Query: {test_query}")

#     response = agent_executor.invoke({
#         "input": test_query
#     })
    
#     print(response)

#     # Safely get the output and handle potential missing keys
#     output = response.get('output', 'No output generated.')

#     output_filename = "test_agent_response.md"
#     with open(output_filename, "w") as f:
#         f.write("# Agent Test Response\n\n")
#         f.write("## Query\n")
#         f.write("```text\n")
#         f.write(test_query + "\n")
#         f.write("```\n\n")
#         f.write("## Final Answer Output\n")
#         f.write("```markdown\n")
#         f.write(output + "\n")
#         f.write("```\n")
       
        
    
#     print(f"\n--- Test Complete ---")
#     print(f"Agent response saved to '{output_filename}'")
