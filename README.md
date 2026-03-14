# PDF → Obsidian Wiki Generator

AI pipeline that converts PDFs into structured Obsidian knowledge bases.

## Features

- PDF text extraction
- automatic concept indexing
- semantic chunk retrieval
- AI wiki generation
- incremental vault updates
- flashcard generation

## Pipeline

PDF
 ↓
Text extraction
 ↓
Concept index generation
 ↓
Chunk retrieval
 ↓
Wiki generation

## Models Supported

This works with the Anthropic API, as well as with any OpenAI-compatible inference endpoint.
- Claude API
- any OpenAI-compatible inference endpoint

## Setup

pip install -r requirements.txt
cp .env.example .env
python main.py

This project does not distribute any model weights.
Users must download or access the model/API separately and agree to the license terms of that service.