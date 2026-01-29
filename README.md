# Task 5: Auto Tagging Support Tickets

## Objective
Automatically tag support tickets into predefined categories using embeddings and LLM.

## Methodology
- Used sentence-transformers for embedding tickets and categories
- Zero-shot tagging with cosine similarity
- Few-shot tagging using DistilGPT2 LLM and prompt engineering
- Compared zero-shot vs few-shot predictions

## Key Results
- Zero-shot tagging correctly predicted top 3 tags for majority of tickets
- Few-shot LLM slightly improved results
- Approach can scale to larger ticket datasets or fine-tuned models
