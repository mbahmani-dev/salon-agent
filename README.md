# 🤖 Persian AI Agent (Multi-Business)

> A production-ready AI Agent engine for small businesses (Salon, Restaurant, Clinic) fully localized for the Persian language and Jalali calendar.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-000000?style=flat)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is this?

This is not just a chatbot. It is an **Agentic AI Engine** designed to handle real-world customer interactions. It reads catalogs, checks real-time availability, books appointments, and escalates complex issues to humans.

It uses a **Provider-Agnostic Architecture**, meaning it works flawlessly on local models (**Ollama**) or cloud models (**OpenAI GPT, Anthropic Claude**) without changing the core business logic.

## ✨ Key Features

- 🗣️ **Persian-First NLP:** Handles Persian digits, Arabic character variants, and zero-width non-joiners natively.
- 📅 **Jalali Calendar Integration:** Automatically resolves relative dates ("فردا", "پنجشنبه") to Gregorian dates for booking.
- 🧠 **RAG (Retrieval-Augmented Generation):** Connects to a local ChromaDB vector database to answer domain-specific questions.
- 🛠️ **Advanced Tool Calling:** 6 core tools (Search Catalog, Date Resolver, Slot Checker, Booking, Human Handoff, FAQ Search).
- 💳 **Payment Gateway Integration:** Supports deposit payments before finalizing appointments.
- 🤖 **Bale Messenger Bot:** Integrated with the Bale messaging app for customer communication.
- 🛡️ **Security Guards:** Prevents the AI from executing unauthorized scripts or booking with empty data.

## 🏗️ Architecture

```text
Customer (Persian) ──▶ Engine (Python) ──▶ LLM (Ollama/GPT/Claude)
                            │
                            ├─▶ Tools (Booking, Catalog, RAG, Dates)
                            │
                            └─▶ Data (SQLite Database, ChromaDB, JSONL logs)
