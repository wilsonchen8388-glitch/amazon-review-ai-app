import streamlit as st
from transformers import pipeline

st.title("Amazon Review AI Analyzer")

st.write("This app analyzes customer reviews using two Hugging Face pipelines.")

sentiment_classifier = pipeline(
    "text-classification",
    model="fabriceyhc/bert-base-uncased-amazon_polarity"
)

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

review = st.text_area(
    "Enter an Amazon product review:",
    "The laptop is fast, but the battery drains quickly and customer support was terrible."
)

if st.button("Analyze Review"):
    sentiment_result = sentiment_classifier(review)[0]

    issue_result = issue_classifier(
        review,
        candidate_labels
    )

    st.subheader("Sentiment Analysis Result")
    st.write("Label:", sentiment_result["label"])
    st.write("Confidence:", round(sentiment_result["score"] * 100, 2), "%")

    st.subheader("Issue Classification Result")
    st.write("Main Issue:", issue_result["labels"][0])
    st.write("Confidence:", round(issue_result["scores"][0] * 100, 2), "%")

    st.subheader("Top Issue Scores")
    for label, score in zip(issue_result["labels"], issue_result["scores"]):
        st.write(label, ":", round(score * 100, 2), "%")
