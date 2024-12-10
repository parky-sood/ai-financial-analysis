
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

st.set_page_config(layout="wide", page_title="Market Insider", page_icon="📈")

def get_stock_info(symbol: str) -> dict:
  """
  Retrives and formats detailed information about a sotck from Yahoo Finance.

  Args:
    symbol (str): The stock ticker symbol to look up.

  Returns:
    dict: A dictionary containing detailed stock information, including ticker,
          name, business summary, city, state, country, industry, and sector.
  """

  data = yf.Ticker(symbol)
  info = data.info

  return {
      "name": info.get("shortName", "N/A"),
      "summary": info.get("longBusinessSummary", "N/A"),
      "sector": info.get("sector", "N/A"),
      "industry": info.get("industry", "N/A"),
      "market_cap": info.get("marketCap", "N/A"),
      "price": info.get("currentPrice", "N/A"),
      "revenue_growth": info.get("revenueGrowth", "N/A"),
      "recommendation": info.get("recommendationKey", "N/A"),
  }

def format_large_number(num):
    if num == "N/A":
        return "N/A"
    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.1f}T"
        elif num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        else:
            return f"${num:,.0f}"
    except:
        return "N/A"

def format_percentage(value):
    if value == "N/A":
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except:
        return "N/A"

# "Buy", "Hold", "Sell", "Strong Buy", "Strong Sell", and sometimes "Underperform" or "Outperform"
def format_recommendation(value):
  match value:
    case "NONE":
      return "N/A"
    case "STRONG_BUY":
      return "BUY"
    case "STRONG_SELL":
      return "SELL"
    case "UNDERPERFORM":
      return "HOLD"
    case "OUTPERFORM":
      return "HOLD"
    
  return value
    

def stock_data_card(data, ticker):

  with st.container():
      st.markdown("""
            
      """, unsafe_allow_html=True)

      st.markdown(f"""
          ### {data['name']} ({ticker})

          **{data['sector']} | {data['industry']}**

          {data['summary'][:60]}...
      """, unsafe_allow_html=True)

      metrics = [
          {"label": "Market Cap", "value": format_large_number(data['market_cap'])},
          {"label": "Price", "value": format_large_number(data['price'])},
          {"label": "Growth", "value": format_percentage(data['revenue_growth'])},
          {"label": "Rating", "value": format_recommendation(data['recommendation'].upper())}
      ]

      for metric in metrics:
        st.metric(label=metric['label'], value=metric['value'], delta=None)

      st.markdown("", unsafe_allow_html=True)

st.title("Market Insider")
st.write("Discover and compare stocks traded on the NYSE")

user_query = st.text_input("Search for stocks by description, sector, or characteristics:")

if st.button("🔍 Find Stocks", type="primary"):
  with st.spinner("Searching..."):
    client = Groq(
      api_key=GROQ_API_KEY
    )

    system_prompt = f"""You are a prompt expert. Convert the user's stock search query into a more searchable format to be like more descriptive like a summary of a company's bussines. This query will be used to search for stocks using embeddings in a vector database and match it with the bussiness summaries of companies in the database. Keep the enhanced query concise. ONLY return the enhanced query nothing else, don't add anything else"""

    llm_response = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[{
                  "role": "system", 
                  "content": system_prompt
                },
                {
                  "role": "user", 
                  "content": f"Convert this stock search query into a searchable format to match the bussines summary of companise , ONLY return the query don't write anything except the query , just the query: {user_query}"
                }]
    )

    enhanced_query = llm_response.choices[0].message.content

    # st.write(f"Enhanced Query: {enhanced_query}")

    # Setup Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "ai-financial-analysis"
    namespace = "stock-descriptions"
    pinecone_index = pc.Index(index_name)

    # Query Pinecone vectors
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(enhanced_query)
    search_results = pinecone_index.query(
      vector=query_embedding.tolist(),
      top_k=5,
      include_metadata=True,
      namespace=namespace
    )

    ticker_list = [item['id'] for item in search_results['matches']]

    stock_data = []
    for ticker in ticker_list:
      data = get_stock_info(ticker)
      if data:
        stock_data.append(data)
    
    cols = st.columns(5)
    for i in range(len(stock_data)):
      with cols[i]:
        stock_data_card(stock_data[i], ticker_list[i])

    # Chart for comparison
    if len(stock_data) > 0:
      st.subheader("Stock Price Comparison")
      fig = go.Figure()

      for i, ticker in enumerate(ticker_list):
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="1y")

        # Normalize the prices to percentage change
        if not hist_data.empty:
          hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100

          fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Normalized'],
            name=f"{ticker}",
            mode='lines'
          ))

      fig.update_layout(
          title="1-Year Price Performance Comparison (%)",
          yaxis_title="Price Change (%)",
          template="plotly_dark",
          height=500,
          showlegend=True
      )

      st.plotly_chart(fig, use_container_width=True)

      with st.spinner("Generating AI Analysis..."):
        stock_info = "\n".join([
              f"Stock: {data['name']} ({ticker_list[i]})\n"
              f"Sector: {data['sector']}\n"
              f"Price: {format_large_number(data['price'])}\n"
              f"Market Cap: {format_large_number(data['market_cap'])}\n"
              f"Growth: {format_percentage(data['revenue_growth'])}\n"
              f"Recommendation: {data['recommendation']}\n"
              f"Summary: {data['summary']}\n"
              for i, data in enumerate(stock_data)
            ])

        chat_prompt = f"""Based on the user's query: "{user_query}"
                          Here's the information about the matching stocks: {stock_info}

                          Please provide a detailed analysis of these stocks, including:
                          1. Why they match the user's query
                          2. Key strengths and potential risks
                          3. Comparative analysis between the stocks
                          4. Investment considerations

                          Format the response in a clear, organized way with sections and bullet points where appropriate.
                        """

        client = Groq(
          api_key=GROQ_API_KEY
        )
        
        chat_response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert stock analyst who always provides detailed, professional analysis."},
                    {"role": "user", "content": chat_prompt}
                ])
        
        analysis = chat_response.choices[0].message.content
        st.subheader("Stock Analysis")
        st.write(analysis)

      st.subheader("Relevant Market News")

      for ticker in ticker_list:
        with st.spinner(f"Loading news for {ticker}..."):
          news_items = newsapi.get_everything(
              q=f"{ticker} stock",
              language='en',
              sort_by='relevancy',
              page_size=2
          )

          if len(news_items["articles"]) > 0:
            st.write(f"**Latest news for {ticker}**")
          for article in news_items['articles']:
            with st.expander(f"{article['title']}"):
              st.write(article['description'])
              st.write(f"**Source:** {article['source']['name']} | **Published on** {article['publishedAt']}")
              st.link_button("Read full article", article['url'])

