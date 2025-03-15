# %% [code] "app.py"
from flask import Flask, render_template, request
import requests
from model_training import load_sentiment_pipeline

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained sentiment model and vectorizer
predict_sentiment = load_sentiment_pipeline(model_path="sentiment_model.pkl", 
                                            vectorizer_path="vectorizer.pkl")

# Set the URL of the GPT-Neo API service (ensure this matches your service settings)
LLM_API_URL = "http://localhost:8000/generate"

def get_llm_summary(text):
    """
    Call the GPT-Neo API to generate a summary for the input text.
    
    Args:
        text (str): The finance-related text to summarize.
        
    Returns:
        str: The generated summary, or an error message if the API call fails.
    """
    payload = {"prompt": f"Summarize this finance-related text:\n\n{text}"}
    try:
        # Send a POST request to the GPT-Neo API with the prompt
        response = requests.post(LLM_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Return the 'summary' field from the response JSON
            return data.get("summary", "[No summary returned]")
        else:
            return f"[Error] GPT-Neo API responded with status: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"[Error] Could not reach GPT-Neo API: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_result = None
    llm_summary = None
    user_text = None
    if request.method == "POST":
        # Retrieve the text input from the form
        user_text = request.form.get("user_text", "")
        # Use the sentiment model to predict the sentiment label
        sentiment_result = predict_sentiment(user_text)
        # Use the GPT-Neo API to generate a summary
        llm_summary = get_llm_summary(user_text)
    # Render the index.html template with the results
    return render_template("index.html",
                           user_text=user_text,
                           sentiment_result=sentiment_result,
                           llm_summary=llm_summary)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
