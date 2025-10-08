
title: RAG Flask Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
RAG Flask Agent with Dialogflow Integration
A Retrieval-Augmented Generation (RAG) agent built with Flask, LangChain, ChromaDB, and HuggingFace embeddings. Integrated with Google Dialogflow for conversational AI.
Features

🧠 RAG-based question answering using ChromaDB vector store
🤗 HuggingFace embeddings for semantic search
📄 Support for text and PDF documents
🔗 Dialogflow webhook integration
⚡ OpenRouter API for LLM completions

API Endpoints
Health Check
GET /
Dialogflow Webhook
POST /webhook
Receives webhook calls from Dialogflow.
Query Endpoint (if you have one)
POST /query
Environment Variables
Set these in your Hugging Face Space settings:

OPENROUTER_API_KEY - Your OpenRouter API key

Local Development
bashpip install -r requirements.txt
python app.py
Dialogflow Setup

Get your Space URL: https://huggingface.co/spaces/[username]/[space-name]
In Dialogflow Console → Fulfillment
Enable Webhook
URL: https://[username]-[space-name].hf.space/webhook
Save and enable webhook in your intents