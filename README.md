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

## Documentation

- Inline architecture docs and docstrings:
	- `src/pdfwiki/main.py` (pipeline orchestration, quality guards, dedupe/skip logic)
	- `src/pdfwiki/retriever.py` (ranking, adaptive context sizing, chunk dedupe)
- Parsing and dedupe regression coverage: `tests/test_main_parsing.py`

## Models Supported

This works with the Anthropic API, as well as with any OpenAI-compatible inference endpoint.
- Claude API
- any OpenAI-compatible inference endpoint

## Setup

pip install -r requirements.txt
cp .env.example .env
python main.py

## Provider Selection

You can switch providers per run using `--provider`:

python src/pdfwiki/main.py notes.pdf --vault ./vault --provider ollama
python src/pdfwiki/main.py notes.pdf --vault ./vault --provider anthropic

If `--provider` is not specified, the app uses `PDF_TO_NOTES_PROVIDER` from environment.

This project does not distribute any model weights.
Users must download or access the model/API separately and agree to the license terms of that service.