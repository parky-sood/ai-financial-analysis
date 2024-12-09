# 📈 Market Insider

[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)](https://huggingface.co/)

Index the New York Stock Exchange (NYSE) by company ticker, characteristics, or name and receive detailed analysis and key market insights about them and similar companies.

## 🎯 Project Overview

### Part 1: Preprocessing NYSE Company Tickers
This project uses a Pinecone database to store an embedding of the company's business summary and index the database by ticker. The Hugging Face `all-mpnet-base-v2` model is used to transform the business summary into a vector embedding. There are 9998 companies on the NYSE. To populate the Pinecone rapidly, a multithreading approach was used to process multiple company tickers and business summaries concurrently. The number of threads in Google Colab were limited to 10 to continue using the free tier resources and preventing API request timeouts from Yahoo Finance for too many requests per minute.

See `Stock-Insider-DB.ipynb` for source code.

### Part 2: Streamlit App Development
When the user queries by sector, business summary, or ticker, a Llama 3.1 LLM is used to enhance the user query to be more usable for searching Pinecone vectors, find the top five most similar companies stored in Pinecone based on cosine similarity, and retrieve them. Further details on these companies are found through the Yahoo Finance API, which provides information about the market cap, industry, sector, analyst recommendations about investing in the company stock, and the past 1 year of stock price history for each company. Another Llama 3.1 LLM is used alongside Prompt Engineering to provide a deep analysis on the stocks using the retrieved information as context. The News API is then used to find internet articles about these companies, and provide further information on the relevant stock market sector.

See `Stock-Insider-App.ipynb` for source code.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/parky-sood/ai-financial-analysis.git
cd ai-finacial-analysis

# Install required packages
pip install -r requirements.txt
```

## 📦 Dependencies

- yfinance
- langchain_pinecone
- groq
- matplotlib
- plotly
- python-dotenv
- langchain-community
- sentence_transformers
- streamlit
- pyngrok
- python-dotenv
- newsapi-python
- ta

## 🚀 Usage

### Running the Web App

Run 
```bash
streamlit run app.py
```
to run locally, or access the hosted application [here](https://market-insider.streamlit.app/).

## 🖥️ Web Interface Features

1. **Natural Language Query for companies:** Enter text describing the company of interest, or refer to the company by name or NYSE ticker. Powered by Llama 3.1 LLM.
2. **Key Company Details:** Relevant search results for companies provided in easy-to-read format with key information about the stocks. Powered by Yahoo Finance API.
3. **Chart of Stock Prices:** Provides stock price changes as a percentage to standardize stock valuations for each relevant company over the past 1 year. Powered by Yahoo Finance API.
4. **Detailed Company Analysis:** Analysis of relevant company history, activities, characteristics to provide investment insights and pros and cons. Powered by Llama 3.1 LLM.
5. **Relevant Market News:** Relevant news about the companies and overall U.S stock market provided with linked articles. Powered by News API.

## 📈 Data Processing

The project includes robust data handling:

- NYSE company tickers from the U.S Securities and Exchange Commission (SEC)
- Pinecone vector storage and Hugging Face embeddings

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Parikshit Sood - parikshitsood.com

