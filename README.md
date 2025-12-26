# Advanced Retrieval Pipeline (RAG)

A **comprehensive and advanced Retrieval-Augmented Generation (RAG) pipeline** built with **pure Python** and modern techniques to significantly improve retrieval quality, without relying on frameworks like LangChain. This project demonstrates the power of **custom-built pipelines using LLMs** for chunking, retrieval, and answer evaluation.

## Features

This project incorporates **state-of-the-art techniques** in modern retrieval systems:

- **Semantic Chunking with LLMs**: Automatically splits documents into meaningful semantic chunks using large language models.  
- **Re-ranking**: Ranks retrieved chunks for higher relevance.  
- **Query Expansion & Rewriting**: Improves retrieval by expanding and refining user queries.  
- **Advanced Evaluations**:  
  - **Chunk-level Evaluation**: Metrics like MRR (Mean Reciprocal Rank), DCG (Discounted Cumulative Gain), and keyword coverage.  
  - **Answer-level Evaluation**: Assess the correctness and relevance of the final generated answer.  
- **App & Evaluator**:  
  - Main pipeline app built with **Gradio** for easy execution.  
  - Evaluator app to visualize retrieval and answer performance.

## Technologies Used

- **Programming Language**: Python (no external RAG frameworks)  
- **APIs & Models**:  
  - GPT models  
  - DeepSeek models  
  - Gemini models  
  - OpenRouter API  
- **Evaluation Metrics**: MRR, DCG, Keyword Coverage, Answer Accuracy  

## Why This Project?

This project is designed for **learning and practical implementation**:

- Demonstrates building a **robust retrieval pipeline from scratch**.  
- Showcases **modern LLM techniques** like semantic chunking, query rewriting, and re-ranking.  
- Enables **evaluation at multiple levels** for fine-grained performance analysis.  
- Fully compatible with **custom company data**, making it production-ready.  
- Ideal for developers and researchers who want to **understand modern RAG systems** without relying on high-level frameworks.

## Conclusion

This Advanced Retrieval Pipeline (RAG) demonstrates the potential of building **highly robust and modern retrieval systems using pure Python and LLMs**. By incorporating techniques such as semantic chunking, query expansion, re-ranking, and multi-level evaluations, this project provides a **comprehensive learning experience** for developers and researchers alike. It is fully adaptable to **custom datasets**, making it a practical solution for real-world applications, while also serving as a **reference for advanced RAG implementations**.

