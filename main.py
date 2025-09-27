import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time
import os
import re
import csv
import json
from urllib.parse import urljoin
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
import warnings
# ADDED: Imports for Web Scraping
import requests
from bs4 import BeautifulSoup

# --- NEW NLP Imports (SpaCy & TextBlob) ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ------------------ Configuration and Globals ------------------
SCRAPED_DATA_PATH = "politifact_data.csv"
COMPARISON_DATA_KEY = "comparison_data"
SELECTED_PHASE_KEY = "selected_phase"

# Load SpaCy and define constants
try:
    # Use the small model for performance
    nlp = spacy.load("en_core_web_sm")
    stop_words = STOP_WORDS
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please ensure it is downloaded.")
    # Fallback in case of environment issue
    stop_words = [] 

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ------------------ Feature Extractors (SpaCy/TextBlob based) ------------------

def lexical_features(X_series):
    """Tokenization + Stopwords removal + Lemmatization (as a single string)."""
    
    @st.cache_data
    def process(text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
        return " ".join(tokens)
        
    X_processed = X_series.apply(process)
    return CountVectorizer().fit_transform(X_processed)

def syntactic_features(X_series):
    """Part-of-Speech tags as features."""
    
    @st.cache_data
    def process(text):
        doc = nlp(text)
        # Use only the coarse-grained POS tags
        return " ".join([token.pos_ for token in doc if token.is_alpha])

    X_processed = X_series.apply(process)
    # Using TF-IDF often performs better with POS tags as features
    return TfidfVectorizer().fit_transform(X_processed)

def semantic_features(X_series):
    """Sentiment polarity & subjectivity from TextBlob as numerical features."""
    
    @st.cache_data
    def process(text):
        blob = TextBlob(text)
        # Scale polarity from -1..1 to 0..1 for easier plotting/comparison
        polarity_scaled = (blob.sentiment.polarity + 1.0) / 2.0
        return [polarity_scaled, blob.sentiment.subjectivity]

    X_features = pd.DataFrame(X_series.apply(process).tolist(),
                              columns=["polarity_scaled", "subjectivity"])
    return X_features.values # Return as NumPy array

def discourse_features(X_series):
    """Sentence count + first word of each sentence as text features."""
    
    @st.cache_data
    def process(text):
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        # Create a document string containing sentence count and first words
        return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

    X_processed = X_series.apply(process)
    return CountVectorizer().fit_transform(X_processed)

def pragmatic_features(X_series):
    """Counts of modality & special words as numerical features."""
    
    @st.cache_data
    def process(text):
        text = text.lower()
        return [text.count(w) for w in pragmatic_words]
    
    X_features = pd.DataFrame(X_series.apply(process).tolist(),
                              columns=[f"Prag_{w.replace('?', 'QM').replace('!', 'EX')}" for w in pragmatic_words])
    return X_features.values # Return as NumPy array

# ------------------ Classifier Factory (from provided NLP code) ------------------

def get_classifiers():
    """Returns a dictionary of ML models for cross-comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "Support Vector Machine": LinearSVC(max_iter=10000, random_state=42, dual='auto'),
        "Naive Bayes Classification": MultinomialNB(),
        "Decision Tree Classification": DecisionTreeClassifier(random_state=42),
    }

# ------------------ ML Execution Functions ------------------

def execute_model_evaluation(model_name: str, X_features, y):
    """Trains and evaluates a single model on the given features."""
    
    models = get_classifiers()
    model = models[model_name]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Get target names from LabelEncoder inverse transform
    le_instance = LabelEncoder().fit(y)
    target_names = le_instance.inverse_transform(np.unique(y_test))
    
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    # Aggregate key metrics (using macro average for multi-class summary)
    macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    
    return {
        'Accuracy': acc,
        'F1-Score (Macro)': macro_f1,
        'Training Time (s)': training_time,
        'cm': json.dumps(cm.tolist()),
        'report_full': json.dumps(report)
    }

@st.cache_data(show_spinner=False)
def analyze_model_performance(nlp_phase: str) -> pd.DataFrame:
    """
    Executes the training and evaluation of ALL 4 ML classifiers across the selected NLP phase.
    """
    
    if not os.path.exists(SCRAPED_DATA_PATH):
        st.error(f"Cannot run analysis: Scraped data file '{SCRAPED_DATA_PATH}' not found.")
        return None 
        
    df = pd.read_csv(SCRAPED_DATA_PATH)
    
    # Pre-processing and split (using hardcoded column names from politifact scrape)
    text_col = "statement"
    label_col = "label"
    
    data = df[[text_col, label_col]].dropna()
    
    SAMPLE_N = min(5000, len(data))
    if SAMPLE_N > 0 and len(data) > SAMPLE_N:
        data = data.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

    X_series = data[text_col].astype(str)
    y_raw = data[label_col]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    if len(np.unique(y)) < 2:
        st.error('Need at least two classes in the target label to train classifiers.')
        return None
    
    # --- 1. FEATURE EXTRACTION based on selected Phase ---
    phase_map = {
        "Lexical": lexical_features,
        "Syntactic": syntactic_features,
        "Semantic": semantic_features,
        "Discourse": discourse_features,
        "Pragmatic": pragmatic_features
    }

    st.warning(f"Extracting features using the **{nlp_phase}** pipeline...")
    X_features = phase_map[nlp_phase](X_series)
    st.success(f"Features extracted. Feature matrix shape: {X_features.shape}")
    
    # --- 2. RUN ALL CLASSIFIERS ---
    model_names = list(get_classifiers().keys())
    agg_results = []
    
    progress_bar = st.progress(0, text=f"Starting model comparison on {nlp_phase} features...")
    total = len(model_names)

    for i, model_name in enumerate(model_names, start=1):
        progress_bar.progress(int(i/total*100), text=f"Training: {model_name} ({i}/{total})")
        
        try:
            res = execute_model_evaluation(model_name, X_features, y)
            res['Model'] = model_name
            agg_results.append(res)
            
        except Exception as e:
            st.error(f'Error while processing model {model_name}: {e}')
            agg_results.append({
                'Model': model_name, 
                'Accuracy': 0.0, 
                'F1-Score (Macro)': 0.0,
                'Training Time (s)': 0.0,
                'error': str(e),
                'cm': '[]',
                'report_full': '{}'
            })
    
    progress_bar.empty()
    return pd.DataFrame(agg_results).set_index('Model')


# ------------------ DATA EXTRACTION (Web Scraper) ------------------
# (The scraping logic remains unchanged from the previous version)

def scrape_data_by_date_range(start_date: date, end_date: date) -> pd.DataFrame:
    """
    WEB SCRAPING FUNCTION: Extracts claims from politifact.com based on the date range.
    """
    base_url = "https://www.politifact.com/factchecks/list/"
    csv_file = SCRAPED_DATA_PATH 
    all_rows = []

    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["author", "statement", "source", "claim_date", "label"])

    current_url = base_url
    stop_scraping = False 

    st.info(f"Scraping claims from {start_date} to {end_date}...")
    placeholder = st.empty()
    page_count = 0

    while current_url and not stop_scraping:
        page_count += 1
        placeholder.text(f"Fetching page {page_count}...")
        
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error during request: {e}. Stopping scrape.")
            break

        page_rows = []

        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date_str, claim_date_obj = None, None

            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+(\d{4}))", date_text)
                if match:
                    claim_date_str = match.group(1)
                    try:
                        claim_date_obj = datetime.strptime(claim_date_str, "%B %d, %Y").date() 
                    except ValueError:
                        continue 
            
            if not claim_date_obj:
                continue

            if claim_date_obj < start_date:
                stop_scraping = True
                break
            
            if start_date <= claim_date_obj <= end_date:
                
                # Statement, Source, Author, Label extraction...
                statement = card.find("div", class_="m-statement__quote").find("a", href=True).get_text(strip=True) if card.find("div", class_="m-statement__quote") and card.find("div", class_="m-statement__quote").find("a", href=True) else None
                source = card.find("a", class_="m-statement__name").get_text(strip=True) if card.find("a", class_="m-statement__name") else None
                
                footer = card.find("footer", class_="m-statement__footer")
                author = None
                if footer:
                    author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                    if author_match:
                        author = author_match.group(1).strip()

                label_img = card.find("img", alt=True)
                label = label_img['alt'].replace('-', ' ').title() if label_img else None

                if statement and label:
                    page_rows.append([author, statement, source, claim_date_str, label])

        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(page_rows)
            all_rows.extend(page_rows)

        if stop_scraping:
            break
            
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            current_url = None 
    
    placeholder.empty()

    if all_rows:
        df = pd.DataFrame(all_rows, columns=["author", "statement", "source", "claim_date", "label"])
        return df
    else:
        return pd.DataFrame(columns=["author", "statement", "source", "claim_date", "label"])


# ------------------ Streamlit UI ------------------

def app():
    st.set_page_config(
        page_title="Model vs NLP Phase Comparator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🤖 Model vs NLP Phase Comparator")
    st.markdown("Benchmarking multiple ML algorithms against a unified NLP feature set from Politifact claims.")

    if COMPARISON_DATA_KEY not in st.session_state:
        st.session_state[COMPARISON_DATA_KEY] = None
    if SELECTED_PHASE_KEY not in st.session_state:
        st.session_state[SELECTED_PHASE_KEY] = "Lexical"

    # ====================================================================
    # LEFT SECTION (Sidebar) - Data Extraction and Control
    # ====================================================================
    with st.sidebar:
        st.header("1. Data & NLP Pipeline Controls")

        # NLP PHASE SELECTION (NEW FOCUS)
        st.subheader("Choose NLP Feature Phase")
        nlp_phase = st.selectbox('Feature Pipeline for all Models:', [
            'Lexical',
            'Syntactic',
            'Semantic',
            'Discourse',
            'Pragmatic'
        ], index=0, key=SELECTED_PHASE_KEY)

        st.markdown("---")
        st.subheader("Data Extraction (Politifact)")
        st.markdown("Define the date range to scrape claims for training.")

        # Date Input Fields
        today = date.today()
        default_start = today - timedelta(days=30)
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date (Minimum)", default_start)
        with col_end:
            end_date = st.date_input("End Date (Maximum)", today)
            
        st.caption(f"Max 5,000 recent claims will be used for training.")

        # Scrape Button
        if st.button("▶️ 1. Run Web Scraper and Save CSV", use_container_width=True, type="primary"):
            if start_date > end_date:
                st.error("Error: Start Date must be before or equal to End Date. The laws of physics apply.")
            else:
                with st.spinner(f"Initiating digital archeology... scraping data between {start_date} and {end_date}..."):
                    scraped_df = scrape_data_by_date_range(start_date, end_date)
                
                if not scraped_df.empty:
                    st.success(f"Scraping complete! {len(scraped_df)} claims extracted and saved to CSV.")
                    st.dataframe(scraped_df.head(), use_container_width=True)
                else:
                    st.warning("No claims found within the specified date range. Try expanding the dates.")

        st.markdown("---")

        # Model Analysis Button
        if st.button("🧠 2. Run Model Comparison", use_container_width=True):
            if not os.path.exists(SCRAPED_DATA_PATH):
                st.error("Please run the Web Scraper first. The algorithms need data!")
            else:
                with st.spinner(f"Running 4 models on the **{nlp_phase}** feature set... this will take a moment."):
                    model_results_df = analyze_model_performance(nlp_phase)
                
                if model_results_df is not None:
                    st.session_state[COMPARISON_DATA_KEY] = model_results_df
                    st.success("Analysis complete! Results ready for display.")


    # ====================================================================
    # MAIN AREA (Center and Right) - Display and Comparison
    # ====================================================================

    st.markdown("---")
    st.header(f"2. Model Comparison on: **{st.session_state[SELECTED_PHASE_KEY]}** Features")

    results_df = st.session_state[COMPARISON_DATA_KEY]

    if results_df is None:
        st.info("👈 Please complete steps 1 and 2 in the sidebar to run the analysis and populate this dashboard.")

    else:
        # Get all metric columns, excluding the stored report/matrix data
        metric_cols = [col for col in results_df.columns if col not in ['cm', 'report_full']]

        col_select, col_empty = st.columns([1, 4])

        with col_select:
            # User selects primary metrics for the center chart
            primary_metrics = st.multiselect(
                "Select Metrics for Center Chart",
                options=[c for c in metric_cols if 'Time' not in c], # Exclude time by default
                default=['Accuracy', 'F1-Score (Macro)']
            )


        col_center, col_right = st.columns([3, 2], gap="large")

        # --- CENTER SECTION (Primary Visualization) ---
        with col_center:
            st.subheader("Model Head-to-Head Performance")
            
            if primary_metrics:
                # Melt the DataFrame to prepare for a grouped/stacked bar chart
                chart_data = results_df[primary_metrics].reset_index().rename(columns={'Model': 'Model Name'}).melt(
                    id_vars='Model Name', 
                    var_name='Metric', 
                    value_name='Value'
                )

                st.bar_chart(
                    chart_data, 
                    x='Model Name', 
                    y='Value', 
                    color='Metric', 
                    height=400
                )
                st.caption(f"Comparing performance across models using **{st.session_state[SELECTED_PHASE_KEY]}** features.")
            else:
                st.warning("Please select at least one metric to visualize in the chart.")


        # --- RIGHT SECTION (Interactive Scatter Plot & Humorous Critique) ---
        with col_right:
            st.subheader("Interactive Trade-off Scatter Plot")
            
            # User selects axes for the scatter plot
            col_x_select, col_y_select = st.columns(2)
            with col_x_select:
                x_axis = st.selectbox("X-Axis (Trade-off)", metric_cols, index=metric_cols.index('Training Time (s)'))
            with col_y_select:
                y_axis = st.selectbox("Y-Axis (Gain)", metric_cols, index=metric_cols.index('Accuracy'))

            st.scatter_chart(
                results_df.reset_index(), # Scatter chart needs columns, not index
                x=x_axis, 
                y=y_axis, 
                color="Model",
                size=50
            )
            st.caption("Find the 'Efficient Genius' Model: low X (cost), high Y (performance).")
            
            st.markdown("---")
            
            # Humorous Model Critique based on Accuracy
            best_model = results_df['Accuracy'].idxmax()
            best_accuracy = results_df['Accuracy'].max()
            best_report = json.loads(results_df.loc[best_model, 'report_full'])

            worst_model = results_df['Accuracy'].idxmin()
            worst_accuracy = results_df['Accuracy'].min()
            
            st.subheader("Model Status Report 😅")
            
            st.markdown(f"""
            **👑 The Champ (Accuracy):** **{best_model}**
            * **Score:** `{best_accuracy:.2%}`
            * **Critique:** "Running on **{st.session_state[SELECTED_PHASE_KEY]}** features, this model is clearly superior. It deserves a raise (or at least better training data)."
            """)
            
            st.markdown(f"""
            **🤡 The Underdog (Accuracy):** **{worst_model}**
            * **Score:** `{worst_accuracy:.2%}`
            * **Critique:** "The **{worst_model}** clearly struggled with the **{st.session_state[SELECTED_PHASE_KEY]}** features. Perhaps it was distracted by a passing squirrel. We recommend rebooting and trying another NLP phase."
            """)
        
        # --- BOTTOM SECTION (Detailed Report for the Best Model) ---
        st.markdown("---")
        st.subheader(f"3. Detailed Metrics: **{best_model}** (The Champion)")
        
        st.metric(label=f"Accuracy for {best_model}", value=f"{best_accuracy:.2%}")
        
        col_report, col_cm = st.columns(2)

        with col_report:
            st.caption("Full Classification Report (Per Class and Averages)")
            report_df = pd.DataFrame(best_report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

        with col_cm:
            st.caption("Confusion Matrix (How the algorithm failed)")
            cm_data = np.array(json.loads(results_df.loc[best_model, 'cm']))
            fig, ax = plt.subplots()
            cax = ax.matshow(cm_data, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            ax.set_title(f'CM for {best_model}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig) # Display the plot

    # File status footer
    if os.path.exists(SCRAPED_DATA_PATH):
        st.divider()
        st.info(f"Input Data File Status: Found at `{SCRAPED_DATA_PATH}`. Last modified: {time.ctime(os.path.getmtime(SCRAPED_DATA_PATH))}")

if __name__ == '__main__':
    app()
