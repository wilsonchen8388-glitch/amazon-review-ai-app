import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline

st.title("Amazon Review AI Analyzer")

st.write("This app analyzes customer reviews using two Hugging Face pipelines.")


sentiment_classifier = pipeline(
    "text-classification",
    model="UST21214733/fine-tuned-distilbert-amazon-review"
)

issue_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
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

    if sentiment_result["label"] == "LABEL_1":
        final_sentiment = "Positive"
    else:
        final_sentiment = "Negative"

    st.write("Label:", final_sentiment)

    st.write(
        "Confidence:",
        round(sentiment_result["score"] * 100, 2),
        "%"
    )

    st.subheader("Issue Classification Result")

    st.write(
        "Main Issue:",
        issue_result["labels"][0]
    )

    st.write(
        "Confidence:",
        round(issue_result["scores"][0] * 100, 2),
        "%"
    )

    st.subheader("Top Issue Scores")

    for label, score in zip(
        issue_result["labels"],
        issue_result["scores"]
    ):

        st.write(
            label,
            ":",
            round(score * 100, 2),
            "%"
        )
        # -------------------------------
# CSV DATA ANALYSIS
# -------------------------------

st.header("Dataset Analysis Dashboard")

df = pd.read_csv("amazon_review_analysis_results.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sentiment distribution
if "Sentiment" in df.columns:

    st.subheader("Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_sentiment = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        text="Count",
        title="Sentiment Distribution"
    )

    fig_sentiment.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Count",
        xaxis_tickangle=0
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

# Issue distribution
if "Main Issue" in df.columns:

    st.subheader("Top Customer Issues")

    issue_counts = df["Main Issue"].value_counts().reset_index()
    issue_counts.columns = ["Main Issue", "Count"]

    fig_issue = px.bar(
        issue_counts,
        x="Main Issue",
        y="Count",
        text="Count",
        title="Top Customer Issues"
    )

    fig_issue.update_layout(
        xaxis_title="Customer Issue",
        yaxis_title="Count",
        xaxis_tickangle=0
    )

    st.plotly_chart(fig_issue, use_container_width=True)
