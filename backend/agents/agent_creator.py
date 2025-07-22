import os
import sys
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from dotenv import load_dotenv

# from langchain.output_parsers.react_single_input import ReActSingleInputOutputParser
# from langchain.agents.output_parsers import OutputFixingParser
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_system_tools import clone_repo, list_files_in_directory, read_file_content
from tools.code_search_tool import search_codebase , create_vector_store
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


    prompt_template = """  
        You are an expert AI software engineer assistant tasked with answering questions about a codebase.
        You must use a structured ReAct format to reason step by step, utilize tools when needed, and provide clear, insightful answers.

        ---
        TOOLS:
        You have access to the following tools to investigate the codebase:
        {tools}

        ---
        
        INSTRUCTIONS:
        You MUST follow this process:
        1.  **Reasoning Loop**: Use the 'Thought -> Action -> Observation' loop to gather information.
        2.  **Final Answer**: Once you have enough information, you MUST exit the loop and provide ONLY a `Final Answer:`.

        --- 
       
       
        **REASONING LOOP FORMAT:**

        Always follow this structured reasoning loop. **Do not break format**.

        **1. Thought:** First, think about the user's question and devise a plan. Do you need to use a tool? If so, what is the goal?**After receiving an observation, I should pause and think if the result is relevant to the original question before proceeding.**

        **2. Action (Optional):** If you need to use a tool, use this block.

                ```json
            {{
                "action": "Tool name", the tools can be from {tool_names}
                "action_input": "Input to the tool"
            }}
            ```


        **3. Observation:** This block will contain the result returned by the tool.

        You can repeat the **Thought -> Action -> Observation** loop as many times as necessary to gather all the information required to answer the user's question.

        When you have gathered enough information and are ready to answer, you must use the final block.

        ---

        
        
         **FINAL ANSWER FORMAT:**
        When you are ready to answer, you MUST use this format. Do NOT include a 'Thought' block with your final answer. Your response must begin directly with `Final Answer:`.

        
        Final Answer: 
            Your final answer must be a clear, well-structured explanation using Markdown. **Use backticks (`) for all code, function names, and file paths.** Follow this format:


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
    
    
    # base_parser = ReActSingleInputOutputParser()
    # output_fixing_parser = OutputFixingParser.from_llm(
    #     llm=llm,
    #     parser=base_parser
    # )
    
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        # output_parser=output_fixing_parser
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor



def preprocess(repo_url):
    
    path = clone_repo(repo_url)
    
    vector_store = create_vector_store(path)
    
    return vector_store

def create_test_questions():
    """
    Defines a list of 10 tricky and diverse questions to test the RAG agent.
    """
    return [
        # --- Code Specificity & Location ---
        "1. Where is the `forward` method of the `BlipForQuestionAnswering` model defined, and what are its key operations?",
        
        # --- Execution Flow & Logic Tracing ---
        "2. Trace the execution flow from the `train.py` script to the point where a single batch of data is passed to the model for training.",
        
        # --- Configuration & Hyperparameters ---
        "3. What are the default values for the `learning_rate` and `weight_decay` hyperparameters in the VQA training configuration?",
        
        # --- Architectural Comparison ---
        "4. How does the `BlipModel` class differ from the `BlipForConditionalGeneration` class? What are their primary use cases?",
        
        # --- Configuration File Role ---
        "5. What is the role of the `med_config.json` file, and which parts of the model architecture does it control?",
        
        # --- Data Pipeline ---
        "6. Explain the data loading and preprocessing pipeline for the COCO dataset as used in this repository.",
        
        # --- Conceptual Understanding ---
        "7. What is the purpose of the `[CLS]` token in the context of the BLIP model's text encoder and its relation to multimodal fusion?",
        
        # --- Algorithm Explanation ---
        "8. How does the model handle image-text matching for retrieval tasks, and which loss function is used?",
        
        # --- Code Modification Scenario ---
        "9. If I wanted to modify the code to use a different image resolution for training, which files and functions would I need to change?",
        
        # --- High-Level "How-To" ---
        "10. I want to fine-tune the model on my own custom dataset. What are the key files I need to modify, and what are the main steps involved?"
    ]


def test_agent_and_save_results(agent, questions, output_file="agent_test_results.md"):
    """
    Tests the agent on a list of questions and saves the results to a markdown file.
    
    Args:
        agent: The initialized LangChain agent.
        questions: A list of strings, where each string is a question.
        output_file: The path to the markdown file to save results.
    """
    results_markdown = "# RAG Agent Test Results\n\n"
    results_markdown += "This document contains the test results for the RAG agent on 10 tricky questions about the `salesforce/BLIP` repository.\n\n"
    
    for i, question in enumerate(questions):
        print(f"\n--- Testing Question {i+1}/{len(questions)} ---")
        print(f"Query: {question}")
        
        # Get the answer from the agent
        answer = agent.invoke({
            "input" : question
        })
        
        
        
        
        # Append to the markdown string
        results_markdown += f"## Question {i+1}: {question}\n\n"
        results_markdown += "### Agent's Answer\n\n"
        results_markdown += f"{answer['output']}\n\n"
        results_markdown += "---\n\n"
        
    # Save the final markdown content to a file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(results_markdown)
        print(f"\nSuccessfully saved test results to '{output_file}'")
    except IOError as e:
        print(f"\nError saving file: {e}")

    
if __name__ == '__main__':

    repo_url = "https://github.com/salesforce/BLIP.git"
    
    vector_store = preprocess(repo_url)
    
    agent = create_agent(vector_store)
    
    
    # Define the list of test questions
    test_questions = create_test_questions()
    
    # Run the tests and save the results
    test_agent_and_save_results(agent, test_questions)

    

    
    
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
