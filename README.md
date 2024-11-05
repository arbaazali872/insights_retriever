# News Research Tool

The **News Research Tool** is a simple yet powerful application designed to streamline news research and question answering. Built on Streamlit, this tool helps users quickly pull relevant insights from multiple articles, perfect for tasks like fact-checking, research, or simply understanding a topic across different sources.

## Why This Tool?

With endless news articles published every day, finding specific information can be like searching for a needle in a haystack. If you’re tired of sifting through pages of content or need quick answers from a mix of sources, this tool can help. Here’s how it works:

1. **Loads and processes article content from URLs you enter.**
2. **Breaks down the information into manageable chunks for analysis.**
3. **Answers your questions directly by pulling relevant insights.**

No need for manual skimming—just input URLs, ask questions, and get answers.

---

## Features

- Load content from up to 3 news URLs.
- Split large articles into chunks for efficient processing.
- Store and retrieve document embeddings with FAISS.
- Answer questions with context provided by a BERT-based question-answering model.
- Display unique sources for verification.

## Getting Started

### Prerequisites

- **Python 3.8+**
- Clone the Repository:
  ```bash
   git clone https://github.com/arbaazali872/insights_retriever.git
   cd <your-repo-directory>
- Create a Virtual Environment (recommended):
  
  ```python3 -m venv venv ```
  
  ```source venv/bin/activate```  # For macOS/Linux
  
  ```.\venv\Scripts\activate```   # For Windows
- Install dependencies:
  
  ```pip install -r requirements.txt```
- Run the Application:
  
  ```streamlit run app.py```
