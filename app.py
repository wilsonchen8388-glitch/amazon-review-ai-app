import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

st.title("Amazon Review AI Analyzer")

st.write("This app analyzes Amazon product reviews using Hugging Face pipelines.")

# Load fine-tuned sentiment model
model_path = "./fine_tuned_distilbert_amazon"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

sentiment_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

# Zero-shot classification pipeline
issue_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

candidate_labels = [
    "Battery Issue",
    "Performance Issue",
    "Product Quality Issue",
    "Delivery Problem",
    "Customer Service Issue",
    "Pricing Issue"
]

review = st.text_area("Enter Amazon Review")

if st.button("Analyze Review"):

    sentiment_result = sentiment_classifier(review)[0]

    issue_result = issue_classifier(
        review,
        candidate_labels
    )

    sentiment_label = sentiment_result["label"]

    if sentiment_label == "LABEL_1":
        final_sentiment = "Positive"
    else:
        final_sentiment = "Negative"

    sentiment_score = round(
        sentiment_result["score"] * 100,
        2
    )

    main_issue = issue_result["labels"][0]

    issue_score = round(
        issue_result["scores"][0] * 100,
        2
    )

    st.subheader("Analysis Result")

    st.write("Sentiment:")
    st.success(final_sentiment)

    st.write("Confidence:")
    st.write(f"{sentiment_score}%")

    st.write("Main Issue:")
    st.warning(main_issue)

    st.write("Issue Confidence:")
    st.write(f"{issue_score}%")