# RepoGraph

RepoGraph is an advanced AI assistant designed to help developers understand and navigate complex software projects. By providing a GitHub URL, you can activate an agent that analyzes the entire codebase, answers architectural questions, and generates dynamic diagrams to visualize code and data flows.

---

## 🚀 Project Setup & Installation

### 1. Create a Conda Environment

We recommend using Python 3.10 or higher.

```bash
conda create --name repograph python=3.11
```

### 2. Activate the Environment

```bash
conda activate repograph
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `backend` directory, and add your OpenAI API key:

```env
OPENAI_API_KEY="sk-YourSecretApiKeyGoesHere"
```

---

