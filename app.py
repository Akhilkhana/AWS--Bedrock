import json
import os
import sys
import boto3
import streamlit as st

# BedrockEmbeddings converts text into numerical vectors using AWS Titan model
# BedrockLLM is used to call Claude or other LLMs hosted on AWS Bedrock
from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM

# RecursiveCharacterTextSplitter breaks large documents into smaller chunks for processing
# PyPDFDirectoryLoader loads all PDF files from a given directory
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# FAISS is a local vector database that stores and searches document embeddings efficiently
from langchain_community.vectorstores import FAISS

# PromptTemplate formats the question + context before sending to the LLM
# RetrievalQA chains the retriever (FAISS) and the LLM into a single pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Create the AWS Bedrock client (uses default credentials from ~/.aws or IAM role)
bedrock = boto3.client(service_name="bedrock-runtime")

# Initialize the embeddings model — Titan converts text to vectors (1536 dimensions)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion():
    """
    Loads all PDFs from the 'data/' folder and splits them into chunks.
    Returns a list of Document objects ready for embedding.
    """
    # Load every PDF file found inside the 'data' directory
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split documents into chunks of 10,000 characters with 100-character overlap
    # Overlap ensures that context at chunk boundaries is not lost
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    """
    Converts document chunks into embeddings and saves the FAISS index to disk.
    The saved index can be reloaded later without re-embedding the documents.
    """
    # Build FAISS vector store by embedding each document chunk
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)

    # Persist the index to a local folder called 'faiss_index'
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    """
    Initializes and returns the Claude Haiku LLM via AWS Bedrock.
    max_tokens limits the length of the model's response.
    """
    llm = BedrockLLM(
        model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512}
    )
    return llm


# This prompt instructs the LLM to answer only from the provided context.
# {context} is filled with the top-k retrieved document chunks.
# {question} is filled with the user's query.
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}
Assistant:"""

# Bind the template to its expected input variables
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    """
    Runs a retrieval-augmented generation (RAG) query:
      1. Retrieves the top-3 most similar document chunks from FAISS.
      2. Feeds those chunks + the user question into Claude via the prompt template.
      3. Returns Claude's answer as a string.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' = concatenate all retrieved chunks into one prompt
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # retrieve the 3 most relevant chunks
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]


def main():
    """
    Streamlit UI entry point.
    - Sidebar: button to ingest PDFs and build/update the FAISS vector store.
    - Main area: text input for the user's question and a button to get Claude's answer.
    """
    st.set_page_config("chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    # Text box where the user types their question
    user_question = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        # Clicking this button reads PDFs, embeds them, and saves the FAISS index
        if st.button("Vector Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    # Clicking this button loads the saved FAISS index and queries Claude
    if st.button("Claude Output"):
        with st.spinner("Processing.."):
            # Load the previously saved FAISS index from disk
            # allow_dangerous_deserialization=True is required by FAISS for local pickle files
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_claude_llm()

            # Display the LLM's answer in the Streamlit app
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


# Standard Python entry point — runs main() when the script is executed directly
if __name__ == "__main__":
    main()
