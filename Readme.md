# Banks (Currently HDFC) Insurance Assistant - Hackathon README

## Overview
The HDFC Insurance Assistant is a Flask-based web application developed for a hackathon, addressing multiple themes: **Customer Support GenAI Agent**, **Insurance Claim Assistant Agent**, **Field Sales Training Agent**, and **Insurance Product Recommendation Agent**. It introduces an innovative **PDF Viewer RAG Agent** to help users understand specific insurance policies and includes **voice chat support** for enhanced user interaction. The application supports two agents: a **Consumer Assistant** for end-users and a **Field Assistant** for employees pitching policies, leveraging Retrieval-Augmented Generation (RAG) for intelligent query handling.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/hdfc-insurance-assistant.git
   cd hdfc-insurance-assistant
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Environment Variables**:
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```
5. **Run the Application**:
   ```bash
   python app.py
   ```
   Access at `http://localhost:8080`.

Else:
Run the provided 3bat files one by one it will do this steps automatically (**note Step2.bat** can take time for installation)

## Hackathon Themes Addressed
The project tackles the following hackathon themes:
1. **Customer Support GenAI Agent**: Provides conversational support for general insurance queries, policy details, and customer assistance.
2. **Insurance Claim Assistant Agent**: Answers queries related to insurance claims by retrieving relevant policy information.
3. **Field Sales Training Agent**: Supports field agents with tailored prompts to pitch policies effectively.
4. **Insurance Product Recommendation Agent**: Recommends suitable insurance products based on user queries and context.
5. **Innovative Idea - PDF Viewer RAG Agent**: Enables users to query specific policy documents interactively, enhancing comprehension of complex insurance policies.

## Agents
The application features two agents:
- **Consumer Assistant**:
  - Handles **Customer Support GenAI**, **Insurance Claim Assistant**, and **Insurance Product Recommendation** tasks.
  - Assists end-users with general inquiries, claim-related questions, and product recommendations using policy document context.
  - Supports **voice chat** for conversational queries, enhancing accessibility (Google Speech Recognition).
    
    ![image](https://github.com/user-attachments/assets/74fab899-8177-49a1-bda5-e41819523ee2)


- **Field Assistant**:
  - Designed for employees pitching insurance policies.
  - Uses specialized prompts (`Field_agent_prompt`) for sales scenarios.
  - ![image](https://github.com/user-attachments/assets/60b6fa52-a024-46f9-a20b-1201a9383f98)


## Voice Chat Support
- **Feature**: Voice chat support enables users to interact with the Consumer Assistant via voice input.
- **Implementation**: Integrated through Google Speech Recognition library
- **Functionality**: Users can ask insurance-related questions or query specific policies using voice, with responses generated based on retrieved documents and conversation context.

## PDF Viewer RAG Agent
- To help users understand specific insurance policies
 ![image](https://github.com/user-attachments/assets/88f62c64-f88a-4632-bdbe-c1eabe4384b4)


Archietecture Diagram <details> <summary>Preview image </summary> ![image](https://github.com/user-attachments/assets/bb6bebeb-c644-4610-a85e-9d971022854f) </details>



## Embedding Model
- **Model Used**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Purpose**: Encodes text into dense vector embeddings for semantic similarity searches.
- **Details**: Lightweight and efficient, runs on CPU or GPU (CUDA-enabled if available) to encode documents and queries.

## Language Model (LLM)
- **Primary LLM**: Groq (`gemma2-9b-it`)
  - Generates contextual responses using retrieved documents.
  - Configured with a temperature of 0.0 for deterministic outputs and a maximum of 1000 tokens.

## Search Techniques
The application employs a **hybrid search** approach for efficient document retrieval:
- **Hybrid Retriever**:
  - **BM25 (Keyword-Based Search)**:
    - Uses the `rank_bm25` library for keyword-based retrieval.
    - Tokenizes documents and queries to compute relevance based on term frequency and inverse document frequency.
  - **Vector Search**:
    - Uses SentenceTransformer embeddings stored in ChromaDB collections (e.g., `pension_plans`, `ulip_plans`, `health_plans`, `all_policies`).
    - Computes cosine similarity between query and document embeddings for semantic relevance.
- **Reranking**:
  - **Model**: CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
  - **Process**: Combines and deduplicates results from BM25 and vector search, then reranks using CrossEncoder to prioritize the top 5 most relevant documents.
- **ChromaDB Collections**:
  - Organizes documents into predefined collections for specific insurance types, enabling targeted retrieval.

## Workflow
The query processing workflow handles both consumer and field agent interactions:
1. **Query Routing**:
   - A `router_system_prompt` determines the relevant insurance collection (e.g., `pension_plans`, `ulip_plans`, `all_policies`, or `general`) based on the query and conversation history.
   - Example from code:
     ```python
     router_system_prompt = "Determine the appropriate insurance collection (pension_plans, ulip_plans, protection_plans, health_plans, savings_plans, annuity_plans, all_policies, general) based on the query and conversation history: {history}"
     ```
   - The LLM evaluates the query and history, defaulting to `all_policies` if no specific match is found.
2. **Document Retrieval**:
   - The `HDFCHybridRetriever` fetches relevant documents using BM25 and vector search, followed by reranking with CrossEncoder.
   - Top 3 documents are included in the LLM prompt.
3. **Response Generation**:
   - Selects `hdfc_agent_system_prompt` (Consumer Assistant) or `Field_agent_prompt` (Field Assistant).
   - The LLM generates a response using the query, retrieved documents, and conversation history.
   - Voice queries are processed similarly, with responses delivered via text or voice output.
4. **Conversation History**:
   - Each chat session is isolated with a unique `chat_id`, stored in `conversation_history/` as JSON files.
   - History informs routing decisions and maintains context.
5. **PDF Viewer RAG Agent**:
   - Users can upload and query specific policy PDFs.
   - PDFs are processed using `PyPDFLoader` and `CharacterTextSplitter`, with embeddings stored in a `SentenceTransformerVectorStore`.
   - Queries are answered by retrieving relevant chunks and generating responses via the LLM.

## Key Features
- **PDF Management**: Upload, organize, and view PDFs in folders/subfolders.
- **Chat Interface**: Supports Consumer and Field Assistant modes with isolated histories and voice chat support.
- **PDF Viewer RAG**: Query specific policy documents for detailed insights.
- **Query History**: Stores queries and responses in TinyDB.
- **Note Creation**: Generate PDF notes for documentation.
