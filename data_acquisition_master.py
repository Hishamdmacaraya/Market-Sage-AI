# %% [code] "data_acquisition_master.py"
import feedparser  # Library to parse RSS feeds
import pandas as pd  # For creating and managing DataFrames

# Function to fetch news from Yahoo Finance
def fetch_yahoo_finance():
    # RSS feed URL for Yahoo Finance (example: AAPL headlines)
    rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)  # Parse the RSS feed
    news_items = []
    # Loop over each entry in the feed
    for entry in feed.entries:
        title = entry.title  # Get the article title
        # Get the summary if it exists; otherwise, use an empty string
        summary = entry.summary if hasattr(entry, 'summary') else ""
        # Combine title and summary to form full text
        text = title + ". " + summary
        # Append the news article as a dictionary with a default sentiment label "neutral"
        news_items.append({"text": text, "label": "neutral", "source": "Yahoo Finance"})
    return news_items

# Function to fetch news from Reuters Business News RSS feed
def fetch_reuters_finance():
    rss_url = "http://feeds.reuters.com/reuters/businessNews"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries:
        title = entry.title
        summary = entry.summary if hasattr(entry, 'summary') else ""
        text = title + ". " + summary
        # Specify source as Reuters
        news_items.append({"text": text, "label": "neutral", "source": "Reuters"})
    return news_items

# Function to fetch news from CNBC RSS feed
def fetch_cnbc_finance():
    rss_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries:
        title = entry.title
        summary = entry.summary if hasattr(entry, 'summary') else ""
        text = title + ". " + summary
        news_items.append({"text": text, "label": "neutral", "source": "CNBC"})
    return news_items

# Function to fetch news from MSN Finance RSS feed
def fetch_msn_finance():
    # Example URL for MSN Money RSS feed (adjust URL if needed)
    rss_url = "https://www.msn.com/en-us/money/rss"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries:
        title = entry.title
        summary = entry.summary if hasattr(entry, 'summary') else ""
        text = title + ". " + summary
        news_items.append({"text": text, "label": "neutral", "source": "MSN Finance"})
    return news_items

# Function to fetch finance-related news from Google News (proxy for Google Finance)
def fetch_google_finance():
    # Google News RSS feed URL for finance search query
    rss_url = "https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries:
        title = entry.title
        summary = entry.summary if hasattr(entry, 'summary') else ""
        text = title + ". " + summary
        news_items.append({"text": text, "label": "neutral", "source": "Google News"})
    return news_items

# Function to merge all articles from the different sources
def merge_all_articles():
    # Call each individual fetch function
    yahoo_articles = fetch_yahoo_finance()
    reuters_articles = fetch_reuters_finance()
    cnbc_articles = fetch_cnbc_finance()
    msn_articles = fetch_msn_finance()
    google_articles = fetch_google_finance()
    
    # Combine all article lists into one
    all_articles = yahoo_articles + reuters_articles + cnbc_articles + msn_articles + google_articles
    return all_articles

# Main execution block: merge articles and save as CSV
if __name__ == "__main__":
    articles = merge_all_articles()  # Get merged list of articles
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(articles)
    # Save DataFrame to CSV file named 'sample_data.csv'
    df.to_csv("sample_data.csv", index=False)
    print("Merged sample_data.csv created with", len(df), "entries")
