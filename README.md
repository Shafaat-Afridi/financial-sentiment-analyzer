# Financial News Sentiment Analyzer

A Streamlit application that analyzes the sentiment of financial news headlines using the FinBERT model, classifying text as positive, negative, or neutral with confidence scores.

## Overview

This application utilizes the ProsusAI/finBERT model, a financial domain-specific version of BERT that has been fine-tuned for sentiment analysis of financial text. The tool allows users to input financial headlines or choose from sample data and receive instant sentiment analysis with probability distributions across sentiment classes.

## Features

- Sentiment analysis specifically calibrated for financial text
- Interactive web interface built with Streamlit
- Visual results with color-coded sentiment and probability charts
- Sample financial headlines for easy testing
- Detailed confidence scores for all sentiment classes
- Informative sections explaining the model and process

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Shafaat-Afridi/financial-sentiment-analyzer.git
   cd financial-sentiment-analyzer
   ```

2. Install required dependencies:
   ```
   pip install streamlit torch transformers pandas numpy
   ```

3. Download the FinBERT model (this will happen automatically on first run)

## Usage

Run the application using Streamlit:
```
streamlit run financial_sentiment_analyzer.py
```

The application will be available at http://localhost:8501 by default.

## How to Use

1. Choose an input method:
   - Enter your own financial headline
   - Select from provided sample headlines

2. Click "Analyze Sentiment" to process the text

3. Review the results:
   - Overall sentiment classification (positive, negative, or neutral)
   - Confidence score for the detected sentiment
   - Bar chart showing probability distribution across all sentiment classes

## How It Works

The application performs sentiment analysis through the following process:

1. **Tokenization**: The financial text is tokenized (split into tokens that the model can process)
2. **Model Processing**: The tokenized text is passed through the FinBERT neural network
3. **Classification**: The model outputs logits which are converted to probabilities for each sentiment class
4. **Results**: The class with the highest probability is selected as the detected sentiment

## Model Information

This application uses the ProsusAI/finBERT model, which is a version of BERT (Bidirectional Encoder Representations from Transformers) that has been fine-tuned specifically for financial sentiment analysis with the following characteristics:

- **Pre-training**: Trained on financial communications and documents
- **Fine-tuning**: Optimized on labeled financial sentences for sentiment classification
- **Classes**: Three-way classification (positive, negative, neutral)
- **Architecture**: Transformer-based model with 110M parameters

## System Requirements

- Python 3.7 or higher
- Minimum 4GB RAM (8GB recommended)
- Internet connection for first run (to download model)
- Approximately 500MB disk space for model storage

## Limitations

- Analysis is limited to text of 512 tokens or fewer
- Results are based on the model's training data and may not reflect current market context
- Performance may vary for highly technical financial terminology
- The model analyzes text in isolation without considering broader market conditions

## Future Development

- Addition of batch processing for multiple headlines
- Integration with real-time financial news APIs
- Support for longer financial documents and reports
- Time-series tracking of sentiment for specific entities
- Export functionality for sentiment analysis results
