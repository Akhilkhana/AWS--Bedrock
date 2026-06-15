# ☁️ AWS Bedrock — Chat with PDF using Claude & LangChain

A Retrieval-Augmented Generation (RAG) application powered by **AWS Bedrock**, **LangChain**, and **Streamlit**. Upload PDF documents and ask questions — Claude answers using only your document content, grounded in real source material.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [AWS Setup](#-aws-setup)
- [Installation](#-installation)
- [Usage](#-usage)
- [Sample Questions](#-sample-questions)
- [Configuration](#-configuration)
- [Contributing](#-contributing)

---

## 🧠 Overview

This project shows how to build a **production-ready RAG pipeline** using AWS-native AI services. Instead of relying on a generic LLM response, the app:

1. Loads PDF documents from the local data/ folder
2. Splits them into chunks and converts them into vector embeddings using **Amazon Titan**
3. Stores embeddings in a local **FAISS** vector index (persisted to disk)
4. At query time, retrieves the most relevant chunks and passes them to **Claude (Haiku)** via AWS Bedrock
5. Returns precise, document-grounded answers in the Streamlit UI

---

## ⚙️ How It Works

The pipeline works in these stages:

1. **Document Loading** — PyPDFDirectoryLoader reads all PDF files from the data/ folder.
2. **Text Splitting** — RecursiveCharacterTextSplitter breaks documents into 10,000-character chunks with 100-character overlap.
3. **Embedding** — Amazon Titan Embeddings (amazon.titan-embed-text-v1) converts each chunk into a 1536-dimensional vector.
4. **Vector Store** — FAISS indexes all vectors and persists the index to disk at faiss_index/.
5. **Retrieval** — At query time, the retriever finds the top-3 most relevant chunks by cosine similarity.
6. **Generation** — Retrieved chunks are passed to Claude Haiku (via Bedrock) using a strict context-only prompt template.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Claude Haiku via AWS Bedrock |
| **Embeddings** | Amazon Titan Embed Text v1 |
| **Vector Store** | FAISS (local, persisted to disk) |
| **Orchestration** | LangChain |
| **AWS SDK** | Boto3 |
| **UI** | Streamlit |
| **Document Source** | PDF files in data/ |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
AWS--Bedrock/
├── app.py               # Main Streamlit application & RAG pipeline
├── data/                # Source PDF documents
│   ├── Attention.pdf    # "Attention Is All You Need" paper
│   └── LLM.pdf          # Large Language Models reference
├── faiss_index/         # Auto-generated: persisted FAISS vector index
├── .env                 # Optional: AWS credentials (if not using IAM)
└── README.md            # This file
```

---

## ✅ Prerequisites

- Python **3.10+**
- An **AWS account** with access to AWS Bedrock
- AWS Bedrock **model access enabled** for:
  - amazon.titan-embed-text-v1
  - anthropic.claude-haiku-20240307-v1:0 (or compatible Haiku version)
- AWS credentials configured (via ~/.aws/credentials, IAM role, or environment variables)

---

## ☁️ AWS Setup

### 1. Enable Bedrock Model Access

1. Open the AWS Console → **Amazon Bedrock** → **Model access**
2. Request access for **Amazon Titan Embeddings** and **Anthropic Claude Haiku**
3. Wait for approval (usually instant for Titan, may take minutes for Claude)

### 2. Configure AWS Credentials

Option A — AWS CLI (recommended):
```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (e.g. us-east-1)
```

Option B — Environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akhilkhana/AWS--Bedrock.git
cd AWS--Bedrock
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your PDFs

Place any PDF files you want to query into the data/ folder.

---

## ▶️ Usage

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**, then:

1. Click **"Vectors Update"** in the sidebar to load PDFs and build the FAISS index (only needed once or when you add new PDFs).
2. Wait for the success message.
3. Type your question in the main input box and click **"Claude Output"**.
4. Claude will retrieve relevant passages and display a grounded answer.

---

## 💬 Sample Questions

Try asking things like (with the included PDFs):

- "What is the attention mechanism in transformers?"
- "Explain the encoder-decoder architecture."
- "What are the key differences between RNNs and transformers?"
- "What is a large language model?"
- "How does self-attention work?"

---

## 🔧 Configuration

You can tune the following parameters in app.py:

| Parameter | Default | Description |
|---|---|---|
| chunk_size | 10000 | Characters per document chunk |
| chunk_overlap | 100 | Overlap between consecutive chunks |
| search_kwargs k | 3 | Number of chunks retrieved per query |
| max_tokens | 512 | Maximum tokens in Claude's response |
| model_id (LLM) | anthropic.claude-haiku-... | AWS Bedrock LLM model ID |
| model_id (Embed) | amazon.titan-embed-text-v1 | AWS Bedrock embedding model ID |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/my-feature)
3. Commit your changes (git commit -m 'Add my feature')
4. Push to the branch (git push origin feature/my-feature)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the MIT License.

---

Built with ❤️ using **AWS Bedrock** · **LangChain** · **Streamlit**
