# RAG Marketing Knowledge Assistant

This project is a Retrieval-Augmented Generation chatbot that helps marketing and analytics teams ask questions from uploaded documents such as campaign notes, FAQs, customer feedback, and reports.

## What This Project Does

- Uploads PDF or TXT documents
- Splits documents into smaller chunks
- Creates embeddings
- Stores embeddings in FAISS vector database
- Retrieves relevant context based on user questions
- Generates grounded answers using an LLM

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- OpenAI API
- PDF/Text document processing

## Business Use Case

Marketing teams often have campaign reports, customer feedback, and internal documentation spread across multiple files. This assistant helps users quickly ask questions and receive context-aware answers from those documents.

## Example Questions

- What were the key campaign performance issues?
- What customer concerns appeared most often?
- What recommendations were mentioned in the report?
- Which marketing channel performed best?

## How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-marketing-knowledge-assistant.git
cd rag-marketing-knowledge-assistant
