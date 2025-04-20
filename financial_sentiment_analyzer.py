import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Set page configuration
# Configure the Streamlit web application with appropriate title and layout settings
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    layout="wide",  # Use wide layout for better visualization of results
    initial_sidebar_state="expanded"  # Start with sidebar expanded for better navigation
)

@st.cache_resource
def load_model():
    """
    Load the FinBERT model and tokenizer using caching to avoid reloading on each interaction
    
    The ProsusAI/finbert model is specifically fine-tuned for financial text sentiment analysis
    with three sentiment classes: positive, negative, and neutral
    
    Returns:
        tuple: (tokenizer, model) The loaded FinBERT tokenizer and model
    """
    # Load pre-trained FinBERT tokenizer - responsible for text preprocessing
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    # Load pre-trained FinBERT model - the neural network that performs sentiment classification
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """
    Analyze the sentiment of the given financial text using the FinBERT model
    
    Args:
        text (str): The financial text to analyze
        tokenizer: The FinBERT tokenizer for preprocessing the text
        model: The FinBERT model for sentiment classification
        
    Returns:
        dict: A dictionary containing sentiment label, confidence score, and detailed scores
    """
    # Preprocess text by tokenizing and preparing tensor inputs for the model
    # padding=True ensures all inputs are the same length
    # truncation=True cuts text that exceeds max_length
    # max_length=512 is the maximum number of tokens the model can process
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Disable gradient calculation for inference to save memory and computation
    with torch.no_grad():
        # Run the model on the preprocessed inputs
        outputs = model(**inputs)
        
    # Convert logits to probabilities using softmax function
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    scores = scores.numpy()[0]  # Convert to numpy array and get first (only) result
    
    # Map numeric indices to human-readable sentiment labels
    # FinBERT specifically uses: 0 = positive, 1 = negative, 2 = neutral
    sentiment_classes = ["positive", "negative", "neutral"]
    
    # Prepare structured result with overall sentiment and detailed scores
    result = {
        "label": sentiment_classes[np.argmax(scores)],  # Get label with highest probability
        "score": float(np.max(scores)),  # Get highest probability value as confidence score
        "scores": {sentiment_classes[i]: float(scores[i]) for i in range(len(sentiment_classes))}  # All scores by label
    }
    
    return result

def load_sample_data():
    """
    Load sample financial news headlines for demonstration purposes
    
    Returns:
        list: A list of sample financial news headlines
    """
    # Provide diverse examples covering different industries and sentiment types
    sample_data = [
        "The GeoSolutions technology will leverage Benefon's GPS solutions by providing Location Based Search Technology.",
        "$ESI on lows, down $1.50 to $2.50 BK a real possibility",
        "For the last quarter of 2010, Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier.",
        "According to the Finnish-Russian Chamber of Commerce, all the major construction companies of Finland are operating in Russia.",
        "$SPY wouldn't be surprised to see a green close",
        "Shell's $70 Billion BG Deal Meets Shareholder Skepticism"
    ]
    return sample_data

def main():
    """
    Main application function handling the Streamlit UI and workflow
    """
    # Load the model and tokenizer with caching
    # This happens only once when the app starts, not on every interaction
    tokenizer, model = load_model()
    
    # Set up the Streamlit UI - title and introduction
    st.title("Financial News Sentiment Analyzer")
    st.write("""
    This application analyzes the sentiment of financial news headlines as positive, negative, or neutral.
    Enter a financial news headline or select from our sample headlines to analyze its sentiment.
    """)
    
    # Create a two-column layout for better organization
    # Left column (col1) for input and results, right column (col2) for information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section - allow users to enter text or choose samples
        st.subheader("Input")
        input_option = st.radio(
            "Choose input method:",
            ("Enter Text", "Choose from Samples")
        )
        
        # Handle different input methods
        if input_option == "Enter Text":
            # Free-form text entry for custom headlines
            text_input = st.text_area("Enter a financial news headline:", height=100)
        else:
            # Selection from pre-defined samples
            sample_data = load_sample_data()
            text_input = st.selectbox("Choose a sample headline:", sample_data)
        
        # Analysis button and processing
        if st.button("Analyze Sentiment"):
            if text_input:
                # Show loading indicator during analysis
                with st.spinner("Analyzing sentiment..."):
                    # Perform sentiment analysis on the input text
                    result = analyze_sentiment(text_input, tokenizer, model)
                
                # Display results section
                st.subheader("Results")
                
                # Define colors for different sentiment types for visual distinction
                sentiment_color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "blue"
                }
                
                # Display the overall sentiment result with appropriate color coding
                # Using HTML/CSS for better formatting and visual appeal
                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; background-color:#f0f2f6;">
                    <h3 style="margin:0;">Detected Sentiment: <span style="color:{sentiment_color[result['label']]};">{result['label'].upper()}</span></h3>
                    <p style="margin:5px 0 0 0;">Confidence: {result['score']:.4f} ({result['score']*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed confidence scores section
                st.subheader("Confidence Scores")
                
                # Create a DataFrame for the scores for easier display
                scores_df = pd.DataFrame({
                    'Sentiment': list(result['scores'].keys()),
                    'Score': list(result['scores'].values())
                })
                
                # Prepare data for the bar chart visualization
                chart_data = pd.DataFrame({
                    'Sentiment': list(result['scores'].keys()),
                    'Score': list(result['scores'].values())
                })
                
                # Sort scores by value for better visualization
                chart_data = chart_data.sort_values('Score', ascending=False)
                
                # Assign colors based on sentiment (not used in current implementation but prepared)
                colors = [sentiment_color[s] for s in chart_data['Sentiment']]
                
                # Display the bar chart showing all sentiment probabilities
                st.bar_chart(chart_data.set_index('Sentiment'))
                
            else:
                # Show warning if no input was provided
                st.warning("Please enter or select a headline to analyze.")
    
    with col2:
        # Information section about the model
        st.subheader("About FinBERT")
        st.write("""
        FinBERT is a pre-trained NLP model for financial sentiment analysis. It was trained on financial communication text and is fine-tuned for sentiment analysis with labeled financial sentences.
        
        The model classifies text as:
        - **Positive**: Indicating favorable financial news
        - **Negative**: Indicating unfavorable financial news
        - **Neutral**: Indicating news with no clear sentiment
        """)
        
        # Information section about how the analysis works
        st.subheader("How It Works")
        st.write("""
        1. The application tokenizes your input text
        2. The tokenized text is fed into the FinBERT model
        3. FinBERT analyzes the sentiment and returns probabilities
        4. The highest probability determines the final sentiment
        """)

# Application entry point
if __name__ == "__main__":
    main()
