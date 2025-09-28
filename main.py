import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt

# --- NLP & ML Imports ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import io

# --- Configuration ---
SCRAPED_DATA_PATH = 'politifact_data.csv'

# --- SpaCy Loading Function (Robust for Streamlit Cloud) ---
@st.cache_resource
def load_spacy_model():
    """Attempts to load SpaCy model, relying on the model being in requirements.txt."""
    try:
        # Load the model. If the requirements.txt fix worked, this will succeed.
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error(f"SpaCy model 'en_core_web_sm' not found. Please ensure the direct GitHub link for the model is correctly listed in your 'requirements.txt' file.")
        st.code("""
        # Example of the line needed in requirements.txt:
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        """, language='text')
        # Re-raise to halt execution if model is critical
        raise e

# Load resources outside main app flow
try:
    NLP_MODEL = load_spacy_model()
except Exception:
    # If loading fails (due to the persistent dependency issue), the app stops here.
    st.stop() 

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# 1. WEB SCRAPING FUNCTION
# ============================

def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Scrapes Politifact fact-checks and filters them by the date range.
    Saves results to SCRAPED_DATA_PATH.
    """
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    
    # Use StringIO to handle CSV writing in memory first
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    
    scraped_rows_count = 0
    page_count = 0
    
    st.caption(f"Starting scrape from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    placeholder = st.empty()

    while current_url and page_count < 100: # Safety limit for pages
        page_count += 1
        placeholder.text(f"Fetching page {page_count}... Scraped {scraped_rows_count} claims so far.")

        try:
            response = requests.get(current_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            placeholder.error(f"Network Error during request: {e}. Stopping scrape.")
            break

        rows_to_add = []
        found_within_date_range = False

        for card in soup.find_all("li", class_="o-listicle__item"):
            # --- Extract Date and Check Range ---
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        # Convert to Pandas Timestamp for easy comparison
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue
            
            if claim_date:
                # Check if date is within the desired range
                if start_date <= claim_date <= end_date:
                    found_within_date_range = True
                    
                    # --- Extract other fields ---
                    statement_block = card.find("div", class_="m-statement__quote")
                    statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block and statement_block.find("a", href=True) else None
                    
                    source_a = card.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None

                    footer = card.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()
                            
                    label_img = card.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img and 'alt' in label_img.attrs else None

                    rows_to_add.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])

                elif claim_date < start_date:
                    # If we encounter a date older than the start date, we can stop scraping
                    placeholder.warning(f"Encountered claim older than start date ({start_date.strftime('%Y-%m-%d')}). Stopping scrape.")
                    current_url = None
                    break # Stop processing cards on this page

        if current_url is None:
            break

        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)

        # Find "Next" page link
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            placeholder.success("No more pages found or last page reached.")
            current_url = None

    placeholder.success(f"Scraping finished! Total claims processed: {scraped_rows_count}")
    
    # Read the data back from StringIO, save to disk, and return DataFrame
    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    
    # Save final cleaned data to CSV
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ============================
# 2. FEATURE EXTRACTION (SPA/TEXTBLOB)
# ============================

def lexical_features(text):
    """Tokenization + Stopwords removal + Lemmatization"""
    doc = NLP_MODEL(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """Part-of-Speech tags (NN, VB, JJ, etc.)"""
    doc = NLP_MODEL(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity & subjectivity using TextBlob"""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count + first word of each sentence (proxy for structure)"""
    doc = NLP_MODEL(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0].lower() for s in sentences if len(s.split()) > 0])}"

def pragmatic_features(text):
    """Counts of modality & special words (must, should, ?, !)"""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# 3. MODEL TRAINING AND EVALUATION
# ============================

def get_classifier(name):
    """Initializes a classifier instance."""
    if name == "Naive Bayes":
        return MultinomialNB()
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
    elif name == "SVM":
        return SVC(kernel='linear', C=0.5, random_state=42)
    return None

def apply_feature_extraction(X, phase, vectorizer=None):
    """Applies the chosen feature extraction technique."""
    if phase == "Lexical & Morphological":
        X_processed = X.apply(lexical_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    
    elif phase == "Syntactic":
        X_processed = X.apply(syntactic_features)
        vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Semantic":
        # Returns a dense DataFrame/Array, no vectorizer needed for sentiment scores
        X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
        return X_features, None

    elif phase == "Discourse":
        X_processed = X.apply(discourse_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Pragmatic":
        # Returns a dense DataFrame/Array, no vectorizer needed for count features
        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
        return X_features, None
    
    return None, None


def evaluate_models(df: pd.DataFrame, selected_phase: str):
    """Trains and evaluates all four models on the given phase features."""
    
    # 1. Data Preparation and Label Encoding
    X_raw = df['statement'].astype(str)
    y_raw = df['label']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    if len(np.unique(y)) < 2:
        st.error("Target column must contain at least two unique classes for classification.")
        return pd.DataFrame() # Return empty DF on failure
    
    # 2. Feature Extraction (Apply to all data once per phase)
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame() # Return empty DF on failure
        
    # 3. Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_features_full, y, test_size=0.25, stratify=y, random_state=42
    )
    
    models_to_run = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42)
    }

    results_list = []
    
    for name, model in models_to_run.items():
        st.caption(f"🚀 Training {name}...")
        
        start_time = time.time()
        
        try:
            # Handle non-sparse data conversion for Naive Bayes if needed
            if name == "Naive Bayes" and not isinstance(X_features_full, (sparse.csr_matrix, sparse.csc_matrix)):
                 X_train_nb = X_train.abs().astype(int)
                 X_test_nb = X_test.abs().astype(int)
            else:
                 X_train_nb = X_train
                 X_test_nb = X_test

            model.fit(X_train_nb, y_train)
            train_time = time.time() - start_time
            
            # Predict
            start_inference = time.time()
            y_pred = model.predict(X_test_nb)
            inference_time = (time.time() - start_inference) * 1000 # ms
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results_list.append({
                "Model": name,
                "Accuracy": acc * 100,
                "F1-Score": f1,
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "Training Time (s)": round(train_time, 2),
                "Inference Latency (ms)": round(inference_time, 2),
                # "Model Object": model # Not saving model objects in final table to keep it light
            })
            
        except Exception as e:
            st.error(f"⚠️ Model Failure: {name} failed on {selected_phase}. Error: {e}")
            results_list.append({
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999, # Placeholder for failed model
            })
            
    return pd.DataFrame(results_list)

# ============================
# 4. HUMOR & CRITIQUE FUNCTIONS
# ============================

def get_phase_critique(best_phase: str) -> str:
    """Humorous critique of the best performing NLP phase."""
    critiques = {
        "Lexical & Morphological": [
            "Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.",
            "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.",
            "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."
        ],
        "Syntactic": [
            "Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.",
            "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?",
            "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."
        ],
        "Semantic": [
            "The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.",
            "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.",
            "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."
        ],
        "Discourse": [
            "Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.",
            "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.",
            "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."
        ],
        "Pragmatic": [
            "The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.",
            "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It’s concise, ruthless, and apparently correct.",
            "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."
        ],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    """Humorous critique of the best performing ML model."""
    critiques = {
        "Naive Bayes": [
            "Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!",
            "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It’s the least complicated tool in the box, which is often the best.",
            "NB pulled off a victory. It’s the 'less-is-more' philosopher who manages to outperform all the complex math majors."
        ],
        "Decision Tree": [
            "The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.",
            "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.",
            "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."
        ],
        "Logistic Regression": [
            "Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.",
            "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.",
            "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."
        ],
        "SVM": [
            "SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.",
            "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.",
            "SVM crushed it! It’s the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."
        ],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))


def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    """Generates a combined, multi-layered critique."""
    
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."

    # Ensure F1-Score is numeric for reliable max finding
    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    
    # Find the best performing model based on F1-Score
    best_model_row = df_results.loc[df_results['F1-Score'].idxmax()]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    
    # Get critiques
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    
    # Combine results and humor
    
    headline = f"👑 The Golden Snitch Award goes to the {best_model}!"
    
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    
    return summary + roast

# ============================
# 5. STREAMLIT APP FUNCTION
# ============================

def app():
    # --- Configuration and Setup ---
    st.set_page_config(page_title='AI vs. Fact: NLP Comparator', layout='wide')

    # --- INTRO SECTION (The Splash Screen) ---
    st.markdown(
        """
        <style>
        .intro-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1f4068 0%, #102d4f 100%);
            border-radius: 15px;
            color: #f7f7f7;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        .intro-header h1 {
            font-size: 3.5em;
            margin-bottom: 0px;
        }
        .intro-header h3 {
            font-size: 1.5em;
            opacity: 0.8;
        }
        </style>
        <div class="intro-header">
            <h1>🧠 The AI Fact-Checking Showdown 🤖</h1>
            <h3>Can algorithms distinguish truth from tweets?</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # --- LAYOUT DEFINITION ---
    col_left, col_center, col_right = st.columns([1, 2, 2])

    # --- State Management ---
    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'df_results' not in st.session_state:
        st.session_state['df_results'] = pd.DataFrame()

    # ============================
    # LEFT COLUMN (Data Input & Controls)
    # ============================
    with col_left:
        st.header("1. Data Sourcing")
        st.subheader("Politifact Time Machine 🕰️")

        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()

        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

        if st.button("Scrape Politifact Data ⛏️"):
            if start_date > end_date:
                st.error("Error: Start Date must be before or equal to End Date.")
            else:
                with st.spinner(f"Initiating digital archeology... scraping claims from {start_date} to {end_date}"):
                    # Convert date objects to Timestamp for the scraper function
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                if not scraped_df.empty:
                    st.session_state['scraped_df'] = scraped_df
                    st.success(f"Scraping complete! {len(scraped_df)} claims successfully harvested.")
                else:
                    st.warning("Scraper returned no data. Try adjusting the date range or checking the target website.")
        
        st.divider()
        st.header("2. Analysis Configuration")
        
        # Selection for NLP Phase
        phases = [
            "Lexical & Morphological",
            "Syntactic",
            "Semantic",
            "Discourse",
            "Pragmatic"
        ]
        selected_phase = st.selectbox("Choose the Feature Set (NLP Phase):", phases, key='selected_phase')
        
        if st.button("Analyze Model Showdown 🥊"):
            if st.session_state['scraped_df'].empty:
                st.error("Please scrape data first!")
            else:
                with st.spinner(f"Engaging {selected_phase} features... training 4 models!"):
                    df_results = evaluate_models(st.session_state['scraped_df'], selected_phase)
                    st.session_state['df_results'] = df_results
                    st.session_state['selected_phase_run'] = selected_phase # Save the phase that was run
                    st.success("Analysis complete! Prepare for the results.")


    # ============================
    # CENTER COLUMN (Metrics & Visuals)
    # ============================
    with col_center:
        st.header("3. Performance Benchmarking")
        
        if st.session_state['df_results'].empty:
            st.info("Awaiting model training. Configure and run the analysis in the left column.")
        else:
            df_results = st.session_state['df_results']
            st.subheader(f"Results for: {st.session_state['selected_phase_run']} Features")

            # Display the main results table
            st.dataframe(
                df_results[['Model', 'Accuracy', 'F1-Score', 'Training Time (s)', 'Inference Latency (ms)']],
                use_container_width=True,
                height=200
            )
            
            # Bar Chart Comparison
            st.divider()
            st.subheader("Metric Comparison")
            
            # Allow user to select metric to plot
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)']
            plot_metric = st.selectbox("Metric to Plot:", metrics, index=1, key='plot_metric_center')
            
            # Plotting the metric
            df_plot = df_results[['Model', plot_metric]].set_index('Model')
            
            # Dynamic color coding for bar chart
            if 'Time' in plot_metric or 'Latency' in plot_metric:
                 st.bar_chart(df_plot, color="#FF5733") # Warning color for high time
            else:
                 st.bar_chart(df_plot, color="#33FF57") # Success color for high score
            
            st.caption(f"Chart shows how each model performed on the selected metric using the **{st.session_state['selected_phase_run']}** features.")


    # ============================
    # RIGHT COLUMN (Critique & Trade-offs)
    # ============================
    with col_right:
        st.header("4. Humorous Critique")
        
        if st.session_state['df_results'].empty:
            st.info("The models are currently in a coffee break. Results coming soon!")
        else:
            # --- Dynamic Humorous Critique ---
            critique_text = generate_humorous_critique(st.session_state['df_results'], st.session_state['selected_phase_run'])
            st.markdown(critique_text)
            st.divider()
            
            # --- Trade-Off Scatter Plot ---
            st.subheader("Speed vs. Quality Trade-off")
            
            metrics_quality = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            metrics_speed = ['Training Time (s)', 'Inference Latency (ms)']
            
            # X-Axis (Speed/Cost) selector
            x_axis = st.selectbox("X-Axis (Speed/Cost):", metrics_speed, key='x_axis', index=1)
            # Y-Axis (Quality) selector
            y_axis = st.selectbox("Y-Axis (Quality):", metrics_quality, key='y_axis', index=0)
            
            # Plotting logic for trade-off (Matplotlib)
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Scatter plot
            ax.scatter(df_results[x_axis], df_results[y_axis], s=150, alpha=0.7)
            
            # Annotate points with model names
            for i, row in df_results.iterrows():
                # Adjusted text position for clarity
                ax.annotate(row['Model'], (row[x_axis] + 0.05 * df_results[x_axis].max(), row[y_axis] * 0.99), fontsize=9)
            
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Model Performance: {x_axis} vs. {y_axis}")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.caption("Look for models in the **bottom-left corner** for the best balance (Low Time, High Quality).")
            
# --- Run App ---
if __name__ == '__main__':
    app()
