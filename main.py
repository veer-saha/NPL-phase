# ============================================
# 📌 Streamlit NLP Phase-wise with All Models
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    """Tokenization + Stopwords removal + Lemmatization"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """Part-of-Speech tags"""
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity & subjectivity"""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count + first word of each sentence"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    """Counts of modality & special words"""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate All Models
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = f"{round(acc, 2)}%"
        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

# ============================
# Streamlit UI
# ============================
st.title("Phase-wise NLP Analysis with Model Comparison")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)

    phase = st.selectbox("Select NLP Phase:", [
        "Lexical & Morphological",
        "Syntactic",
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])

    if st.button("Run Comparison"):
        X = df[text_col].astype(str)
        y = df[target_col]

        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                      columns=["polarity", "subjectivity"])

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Pragmatic":
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                      columns=pragmatic_words)

        # Run all models
        results = evaluate_models(X_features, y)

        # Convert results to DataFrame
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df["Accuracy_float"] = results_df["Accuracy"].str.rstrip('%').astype(float)
        results_df = results_df.sort_values(by="Accuracy_float", ascending=False).reset_index(drop=True)

        # Display results
        st.subheader("Model Comparison Results")
        st.dataframe(results_df[["Model", "Accuracy"]])

        # Bar chart
        acc_values = results_df["Accuracy_float"]
        plt.figure(figsize=(6, 4))
        plt.bar(results_df["Model"], acc_values, alpha=0.7)
        plt.ylabel("Accuracy (%)")
        plt.title(f"Model Performance on {phase}")
        plt.xticks(rotation=30)

        # Add percentage labels
        for i, v in enumerate(acc_values):
            plt.text(i, v + 1, f"{v:.0f}%", ha='center')

        st.pyplot(plt)
