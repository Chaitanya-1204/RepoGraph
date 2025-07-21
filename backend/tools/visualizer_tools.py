def generate_graph(mermaid_syntax):
    """
    A simple passthrough function that acts as a tool for the agent.

    The agent will generate the Mermaid.js syntax based on its analysis,
    and then call this tool to signal that its final output should include
    a visual graph. The backend will then pass this syntax to the frontend
    for rendering.

    Args:
        mermaid_syntax: A string containing valid Mermaid.js graph syntax.

    Returns:
        The exact mermaid_syntax string that was passed in.
    """
    
    return mermaid_syntax