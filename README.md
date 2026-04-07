## Olist E-commerce Chatbot

An AI-powered chatbot for the Olist Brazilian e-commerce dataset, capable of answering natural language questions about orders, products, and customer reviews.

## Features
- SQL Agent: answers data/statistics questions using natural language
- RAG: retrieves customer reviews using FAISS vector search
- Hybrid: combines SQL + RAG for recommendation questions
- LLM fallback: handles general conversation

## Tech Stack
Python, LangChain, OpenAI GPT-4o-mini, FAISS, SQLite, Streamlit

## Dataset
Olist Brazilian E-commerce Dataset (Kaggle)
- 100K+ orders, 40K+ customer reviews
- 8 tables: orders, products, customers, sellers, payments, reviews, geolocation, items

## Architecture
- Router: LLM classifies user intent into SQL / RAG / Hybrid / LLM
- SQL Agent: generates and executes SQL queries on SQLite database
- RAG: FAISS vector store with 40K+ customer reviews, OpenAI embeddings
- Hybrid: combines SQL results with RAG for recommendation queries

## How it works
User question → Router (LLM classifies intent) → SQL / RAG / Hybrid / LLM → Answer

## Example Questions
- "What are the top 10 best selling categories?" → SQL
- "What do customers complain about most?" → RAG
- "I have 100 reais, recommend me a gift with good reviews" → Hybrid
- "Hello, what can you help me with?" → LLM
