# AI-Powered Sports Quiz Generation Agent

An intelligent, interactive system designed to generate sports trivia and quizzes for maximum social media engagement. It utilizes a dual-engine workflow: Retrieval-Augmented Generation (RAG) powered by ChromaDB for legacy structural data, and a live Web Search agent routing tier to catch real-time information.

## 🚀 Key Features Implemented
- **Streamlit Interface:** High-fidelity configuration controls selecting target sports and specific difficulties.
- **RAG Architecture:** Vectorization and indexing layer built on ChromaDB local persistent stores.
- **Agentic Routing:** Dynamic logic splitting query execution loops between historical records and live news lookups.

## 🔧 Installation and Execution Guide
1. Clone the repository and install all required framework libraries:
   ```bash
   pip install -r requirements.txt
