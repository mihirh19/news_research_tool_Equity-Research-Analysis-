![news_research_tool_Equity-Research-Analysis-](https://socialify.git.ci/mihirh19/news_research_tool_Equity-Research-Analysis-/image?description=1&descriptionEditable=RockyBot%20is%20a%20news%20research%20tool%20that%20processes%20and%20analyzes%20news%20articles%20from%20given%20URLs.%20&font=Source%20Code%20Pro&logo=https%3A%2F%2Fgithub.com%2Fmihirh19%2Fnews_research_tool_Equity-Research-Analysis-%2Fassets%2F128199131%2F5146a94d-cadc-4f99-b9de-c8d400c3e938&name=1&pattern=Circuit%20Board&theme=Light)

# FinGuru: News Research Tool 📈

FinGuru is a news research tool that processes and analyzes news articles from given URLs. It leverages LangChain, Google embeddings, and Streamlit to provide insights and answers based on the content of the articles.

## Features

- Fetch and parse news articles from URLs
- Split articles into manageable chunks
- Create embeddings for the text using Cohere
- Store embeddings in a FAISS index for efficient retrieval
- Query the processed data to get answers and sources

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- Google API Key
## Used LLM
`cohere command-r-plus`

## AWS Architecture

![Alt text](images/Frame.png)

# Equity Research Analysis

![Alt text](images/image.png)
![Alt text](images/image-1.png)

### Tech Architecture

    - Issue 1 : Copy pasting article in ChatGPt is tedious
    - Issue 2 : We need an aggregate knowledge base

![Alt text](images/image-2.png)
![Alt text](images/image-3.png)

### Revenue of apple

![Alt text](images/image-5.png)

### calories in apple

![Alt text](images/image-4.png)

`Semantic search`

## Vector Database

![Alt text](images/image-6.png)



## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/mihirh19/news_research_tool_Equity-Research-Analysis-.git
cd news_research_tool_Equity-Research-Analysis-
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install the required packages:

```bash
   pip install -r requirements.txt
```

## Setup

### 1. Create a file named `secrets.toml` in the `.streamlit` directory with the following content:

```toml
COHERE_API_KEY = "your-cohere-api-key"
```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1.  Open the Streamlit application in your browser.
2.  Enter the number of URLs you want to process in the sidebar.
3.  Provide the URLs for the news articles.
4.  Click on "Process URLs" to fetch and analyze the articles.
5.  Enter a query in the text input box and click "Submit" to get answers based on the processed data.

## Example :

1.  enter 3 as number of urls
2.  provide following urls:
    1. https://www.moneycontrol.com/news/business/tata-motors-to-use-new-1-billion-plant-to-make-jaguar-land-rover-cars-report-12666941.html
    2. https://www.moneycontrol.com/news/business/stocks/tata-motors-stock-jumps-x-after-robust-jlr-sales-brokerages-bullish-12603201.html
    3. https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-1188-sharekhan-12411611.html
3.  Click "Process URLs" to start processing.
4.  Enter a query like `what is the target price of tata motors ?` and click `Submit` to get the answer.

## Author

👤 **Mihir Hadavani**

- Twitter: [@mihirh21](https://twitter.com/mihirh21)
- Github: [@mihirh19](https://github.com/mihirh19)
- LinkedIn: [@mihir-hadavani-996263232](https://linkedin.com/in/mihir-hadavani-996263232)

## Show your support

Give a ⭐️ if this project helped you!
