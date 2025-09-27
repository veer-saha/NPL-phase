# ============================================
# 📌 Streamlit NLP Phase-wise Model Comparator
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
from datetime import datetime
import time
import subprocess
import sys
import warnings

# --- NLP & ML Imports ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================
# Global Configuration
# ============================
SCRAPED_DATA_PATH = 'politifact_data.csv'
BASE_URL = "https://www.politifact.com/factchecks/list/"

# SpaCy setup: Rely entirely on requirements.txt for model installation
@st.cache_resource
def load_spacy_model():
    """
    Attempts to load SpaCy model. If missing, it fails, assuming the user 
    must ensure the 'en_core_web_sm' package is installed via requirements.txt.
    """
    model_name = "en_core_web_sm"
    try:
        # 1. Try loading the model directly
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        # If it fails here, it means installation was blocked or the package wasn't listed in requirements.txt
        st.error(f"SpaCy model '{model_name}' could not be loaded.")
        st.info("To fix this, please ensure **`en_core_web_sm`** is added as a separate line in your **`requirements.txt`** file.")
        raise RuntimeError("SpaCy model setup failed. Please check requirements.txt.")


try:
    nlp = load_spacy_model()
except RuntimeError:
    st.stop() # Stop the app if model setup fails

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# Core Functions: Data Scraping
# ============================

def scrape_data_by_date_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Scrapes fact-check data from Politifact within the given date range.
    Stops scraping when a claim is older than the start_date.
    """
    st.subheader("🕸️ Running Web Scraper...")
    
    # 1. Setup
    csv_file = SCRAPED_DATA_PATH
    current_url = BASE_URL
    scraped_rows = []
    
    # Write CSV header once (or start fresh)
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["author", "statement", "source", "date", "label"])

    page_count = 0
    placeholder = st.empty()
    
    while current_url:
        page_count += 1
        placeholder.text(f"Fetching page {page_count}...")

        try:
            response = requests.get(current_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error during request: {e}. Stopping scraper.")
            break

        cards_to_append = []
        found_recent_claim = False
        
        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            
            # Date extraction and filtering
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        date_str = match.group(1)
                        claim_date = datetime.strptime(date_str, "%B %d, %Y")
                    except ValueError:
                        claim_date = None # skip if date format fails

            if not claim_date:
                continue
            
            if claim_date < start_date:
                # Claim is older than the user's start date, stop the entire process
                placeholder.info(f"Reached claims older than {start_date.strftime('%Y-%m-%d')}. Stopping scrape.")
                current_url = None
                break 
            
            if claim_date > end_date:
                # Claim is newer than the user's end date, skip this one
                continue

            # Found a relevant claim within the window
            found_recent_claim = True
            
            # --- Extract fields ---
            statement_block = card.find("div", class_="m-statement__quote")
            statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block else None

            source_a = card.find("a", class_="m-statement__name")
            source = source_a.get_text(strip=True) if source_a else None

            footer = card.find("footer", class_="m-statement__footer")
            author = None
            if footer:
                author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                if author_match:
                    author = author_match.group(1).strip()

            label_img = card.find("img", alt=True)
            label = label_img['alt'].replace('-', ' ').title() if label_img else None

            cards_to_append.append([author, statement, source, claim_date.strftime("%Y-%m-%d"), label])

        # Append rows to CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(cards_to_append)
        
        scraped_rows.extend(cards_to_append)

        if current_url is None:
            break

        # Find "Next" page link
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(BASE_URL, next_href)
        else:
            placeholder.info("No more pages found, scraping done.")
            current_url = None

    placeholder.empty()
    if scraped_rows:
        return pd.DataFrame(scraped_rows, columns=["author", "statement", "source", "date", "label"])
    else:
        return pd.DataFrame()

# ============================
# NLP Phase Feature Extractors
# ============================

def lexical_features(text):
    """Tokenization + Stopwords removal + Lemmatization using SpaCy."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """Part-of-Speech (POS) tags using SpaCy."""
    doc = nlp(text)
    # Combine tokens and their POS tags for vectorization (e.g., 'word_NOUN')
    pos_tags = " ".join([f"{token.text}_{token.pos_}" for token in doc if token.is_alpha])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity & subjectivity using TextBlob."""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count + first word of each sentence using SpaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Get the first token of each non-empty sentence
    first_words = []
    for sent in sentences:
        sent_doc = nlp(sent)
        if sent_doc and sent_doc[0].is_alpha:
            first_words.append(sent_doc[0].text)
            
    # Vectorize sentence count and the first words
    return f"SENT_COUNT_{len(sentences)} {' '.join(first_words)}"

def pragmatic_features(text):
    """Counts of modality & special words."""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Core Functions: Model Evaluation
# ============================

def get_classifier(name: str):
    name = name.lower()
    if 'naive' in name:
        return MultinomialNB()
    elif 'decision' in name or 'tree' in name:
        return DecisionTreeClassifier(random_state=42)
    elif 'logistic' in name:
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    elif 'svm' in name or 'svc' in name:
        return LinearSVC(max_iter=10000, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {name}")


@st.cache_data(show_spinner=False)
def analyze_model_performance(data_df: pd.DataFrame, selected_phase: str) -> pd.DataFrame:
    """
    Processes data using the selected NLP phase and evaluates all ML models.
    Returns a DataFrame comparing model performance.
    """
    st.info(f"🤖 Training models on data using **{selected_phase}** features...")

    # 1. Prepare Data
    X = data_df['statement'].astype(str)
    y_raw = data_df['label']
    
    # Label encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    if len(np.unique(y)) < 2:
        st.error("Target label has fewer than two classes. Cannot train a classifier.")
        return pd.DataFrame()

    # 2. Feature Extraction based on selected phase
    X_processed = X
    X_features = None

    # Time feature extraction
    start_time = time.time()
    
    if selected_phase in ['Lexical & Morphological', 'Syntactic', 'Discourse']:
        # These phases use Count/TFIDF Vectorization
        
        if selected_phase == 'Lexical & Morphological':
            X_processed = X.apply(lexical_features)
        elif selected_phase == 'Syntactic':
            X_processed = X.apply(syntactic_features)
        elif selected_phase == 'Discourse':
            X_processed = X.apply(discourse_features)

        # Use CountVectorizer for Lexical/Syntactic/Discourse features
        try:
            vectorizer = CountVectorizer()
            X_features = vectorizer.fit_transform(X_processed)
        except Exception as e:
            st.error(f"Error during vectorization: {e}")
            return pd.DataFrame()

    elif selected_phase == 'Semantic':
        # Semantic features are numeric (Polarity, Subjectivity)
        X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
        
    elif selected_phase == 'Pragmatic':
        # Pragmatic features are numeric (Counts of modality words)
        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)

    # 3. Handle data split and dimensionality
    # If X_features is a DataFrame (Semantic/Pragmatic), convert to sparse matrix for compatibility with ML models
    if isinstance(X_features, pd.DataFrame):
        X_features = X_features.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Train and Evaluate Models
    results_list = []
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "SVM (Linear SVC)": LinearSVC(max_iter=10000, random_state=42)
    }

    progress_bar = st.progress(0, text="Training models...")
    total_models = len(models)
    
    for i, (name, model) in enumerate(models.items()):
        
        # Training Time
        train_start = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # Inference Time & Metrics
            inference_start = time.time()
            y_pred = model.predict(X_test)
            inference_time = (time.time() - inference_start) * 1000 # to ms

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results_list.append({
                "Model": name,
                "Accuracy": acc,
                "F1-Score": f1,
                "Training Time (s)": round(train_time, 2),
                "Inference Latency (ms)": round(inference_time / len(y_test) * 1000, 2), # ms per sample
                "Report": None # Placeholder for detailed report if needed
            })
        except Exception as e:
            st.error(f"Model '{name}' failed during training: {e}")
            results_list.append({
                "Model": name, "Accuracy": 0.0, "F1-Score": 0.0, "Training Time (s)": 0.0, "Inference Latency (ms)": 0.0, "Report": str(e)
            })

        progress_bar.progress((i + 1) / total_models, text=f"Trained {i+1}/{total_models} models.")

    progress_bar.empty()
    st.success(f"Training complete! Feature extraction time: {round(time.time() - start_time, 2)}s")

    return pd.DataFrame(results_list)

# ============================
# Streamlit App Layout
# ============================

def app():
    st.set_page_config(
        page_title='AI Fact-Checker Benchmarker',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    st.title('Politifact Fact-Checker Benchmark')
    st.caption('Where machine learning models fight over who can spot the most believable lies.')

    # Define paths and columns
    SCRAPED_DATA_PATH = 'politifact_data.csv'
    
    # Initialize session state for data persistence
    if 'scraped_df' not in st.session_state:
        st.session_state.scraped_df = pd.DataFrame()
    if 'model_results_df' not in st.session_state:
        st.session_state.model_results_df = pd.DataFrame()

    # --- 1. Left Section (Sidebar: Controls & Scraping) ---
    with st.sidebar:
        st.header('1. Data Acquisition')
        
        # Date selection for web scraping
        end_date = st.date_input("End Date (Newest Claim):", datetime.today().date())
        start_date = st.date_input("Start Date (Oldest Claim):", datetime(2023, 1, 1).date())

        if st.button('Scrape Data & Generate CSV'):
            # Convert date objects to datetime for function
            dt_start = datetime.combine(start_date, datetime.min.time())
            dt_end = datetime.combine(end_date, datetime.max.time())

            if dt_start > dt_end:
                st.error("Error: Start Date must be before or equal to End Date.")
            else:
                with st.spinner(f"Initiating digital archeology... scraping claims between {start_date} and {end_date}."):
                    scraped_df = scrape_data_by_date_range(dt_start, dt_end)
                    
                    if not scraped_df.empty:
                        st.session_state.scraped_df = scraped_df
                        st.success(f"Scraping complete! {len(scraped_df)} claims ready for analysis.")
                    else:
                        st.session_state.scraped_df = pd.DataFrame()
                        st.warning("Scraping returned no claims in the specified date range.")

        # --- Model Control (Runs only if data is present) ---
        if not st.session_state.scraped_df.empty:
            st.markdown('---')
            st.header('2. Model Benchmarking')
            
            # Choose the NLP Feature Set (Phase)
            phases = [
                "Lexical & Morphological",
                "Syntactic",
                "Semantic",
                "Discourse",
                "Pragmatic"
            ]
            selected_phase = st.selectbox(
                "Select NLP Feature Phase:",
                phases,
                help="Choose which set of language features (e.g., just words, or sentence structure) the ML models will use."
            )

            if st.button('Run Model Comparison'):
                with st.spinner(f"Training 4 ML models on {len(st.session_state.scraped_df)} claims using the '{selected_phase}' feature set..."):
                    results_df = analyze_model_performance(st.session_state.scraped_df, selected_phase)
                    st.session_state.model_results_df = results_df
                st.success(f"Analysis complete for the {selected_phase} phase!")


    # --- 2. Main Columns (Center & Right) ---
    col_center, col_right = st.columns([3, 2])

    # --- Center Section: Model Comparison & Results ---
    with col_center:
        st.header('Center Stage: Model Performance')

        if st.session_state.scraped_df.empty:
            st.info("Start by setting the dates and clicking 'Scrape Data' in the sidebar.")
        else:
            st.subheader(f"Data Summary ({len(st.session_state.scraped_df)} Claims)")
            st.dataframe(st.session_state.scraped_df.head(5), use_container_width=True)
            
            if st.session_state.model_results_df.empty:
                st.warning("Click 'Run Model Comparison' in the sidebar to generate performance metrics.")
            else:
                st.subheader('Battle Report: Model vs. Phase')
                
                # Dynamic Metric Selection for Plotting
                metrics = st.session_state.model_results_df.columns.drop('Model').tolist()
                
                st.session_state['plot_metric'] = st.selectbox(
                    "Metric to Visualize:",
                    metrics,
                    index=metrics.index('Accuracy') if 'Accuracy' in metrics else 0
                )
                
                # 1. Bar Chart Comparison
                plot_df = st.session_state.model_results_df.sort_values(
                    by=st.session_state.plot_metric, 
                    ascending=False
                )
                
                st.markdown(f"#### Bar Chart: {st.session_state.plot_metric}")
                
                # Create Altair/Streamlit chart
                st.bar_chart(
                    plot_df.set_index('Model')[st.session_state.plot_metric], 
                    use_container_width=True,
                    height=350
                )
                
                st.dataframe(st.session_state.model_results_df.sort_values(by='Accuracy', ascending=False), use_container_width=True)


    # --- 3. Right Section: Humorous Analysis & Trade-offs ---
    with col_right:
        st.header('Right Hook: The AI Critique')

        if not st.session_state.model_results_df.empty:
            df_results = st.session_state.model_results_df.copy()
            best_model = df_results.loc[df_results['Accuracy'].idxmax()]
            worst_model = df_results.loc[df_results['Accuracy'].idxmin()]
            
            st.markdown(f"**Current Phase:** `{selected_phase}`")
            st.markdown('---')

            # Humorous Critique
            st.markdown("##### The Judge's Ruling ⚖️")
            st.success(f"""
            **Winner: {best_model['Model']} ({best_model['Accuracy']:.3f} Accuracy)**
            > "The **{best_model['Model']}** clearly understood the assignment. With the '{selected_phase}' feature set, it's operating on a higher plane of factual existence. Give it a raise! (Or at least, don't unplug it.)"
            """)

            if best_model['Model'] != worst_model['Model']:
                st.error(f"""
                **Loser: {worst_model['Model']} ({worst_model['Accuracy']:.3f} Accuracy)**
                > "The poor **{worst_model['Model']}** seems to have mistaken this for a horoscope prediction session. Its performance is a solid 'Need More Coffee' grade. Maybe it needs to re-read the features."
                """)
            
            st.markdown('---')

            # Interactive Trade-off Plot
            st.markdown("##### The Trade-off Chart: Speed vs. Brains")
            
            # Select axis for scatter plot
            metrics = df_results.columns.drop('Model').tolist()
            x_axis = st.selectbox("X-Axis (Effort/Speed):", metrics, key='x_axis', index=metrics.index('Training Time (s)'))
            y_axis = st.selectbox("Y-Axis (Quality):", metrics, key='y_axis', index=metrics.index('Accuracy'))

            # Plotting logic for trade-off (Matplotlib)
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Scatter plot
            ax.scatter(df_results[x_axis], df_results[y_axis], s=100)
            
            # Annotate points
            for i, row in df_results.iterrows():
                # Adjust annotation position slightly to avoid overlap
                ax.annotate(row['Model'].replace(' ', '\n'), (row[x_axis] * 1.02, row[y_axis] * 0.98), fontsize=8)
            
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Model Trade-off: {y_axis} vs. {x_axis}")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            st.info("The ideal model should be in the top-left corner (High Quality, Low Effort/Speed).")

    st.markdown('---')
    st.markdown('**Deployment Notes & Tips**')
    st.markdown(
        """
        - ⚠️ **Web Scraping Caution**: Scraping `politifact.com` may be slow and can be stopped by the website. The app stops automatically when it finds claims older than your start date.
        - **SpaCy Fix**: **You must ensure the dependency is installed correctly.** The application now fails early if the model is missing, guiding you to the correct fix: adding the model package name to `requirements.txt`.
        - **Next Step**: Please update your **`requirements.txt`** file as shown below to resolve this final dependency issue.
        """
    )


if __name__ == '__main__':
    app()
