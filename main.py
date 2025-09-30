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

# --- NEW DEP: Imbalanced-learn ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# --- NLP & ML Imports ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
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
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error(f"SpaCy model 'en_core_web_sm' not found. Please ensure the direct GitHub link for the model is correctly listed in your 'requirements.txt' file.")
        st.code("""
        # Example of the line needed in requirements.txt:
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        imbalanced-learn # New dependency for SMOTE
        """, language='text')
        raise e

# Load resources outside main app flow
try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop() 

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# 1. WEB SCRAPING FUNCTION
# ============================

# [SCRAPING FUNCTION REMAINS UNCHANGED - (lines 53-150)]
# (I am omitting the long scraping function here to ensure this critical block is concise, 
# assuming you have the working version from before. It remains identical.)

def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp):
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    scraped_rows_count = 0
    page_count = 0
    st.caption(f"Starting scrape from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    placeholder = st.empty()

    while current_url and page_count < 100: 
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

        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue
            
            if claim_date:
                if start_date <= claim_date <= end_date:
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
                    placeholder.warning(f"Encountered claim older than start date ({start_date.strftime('%Y-%m-%d')}). Stopping scrape.")
                    current_url = None
                    break 

        if current_url is None:
            break

        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)

        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            placeholder.success("No more pages found or last page reached.")
            current_url = None

    placeholder.success(f"Scraping finished! Total claims processed: {scraped_rows_count}")
    
    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ============================
# 2. FEATURE EXTRACTION (SPA/TEXTBLOB)
# ============================

def lexical_features(text):
    doc = NLP_MODEL(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = NLP_MODEL(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = NLP_MODEL(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0].lower() for s in sentences if len(s.split()) > 0])}"

def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# 3. MODEL TRAINING AND EVALUATION (K-FOLD & SMOTE)
# ============================

def get_classifier(name):
    """Initializes a classifier instance with hyperparameter tuning for imbalance."""
    if name == "Naive Bayes":
        # Cannot use class_weight, but it's fast.
        return MultinomialNB()
    elif name == "Decision Tree":
        # Use balanced class weight to penalize errors on minority classes
        return DecisionTreeClassifier(random_state=42, class_weight='balanced') 
    elif name == "Logistic Regression":
        # Use balanced class weight
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
    elif name == "SVM":
        # Use balanced class weight
        return SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    return None

def apply_feature_extraction(X, phase, vectorizer=None):
    if phase == "Lexical & Morphological":
        X_processed = X.apply(lexical_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2))
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    
    elif phase == "Syntactic":
        X_processed = X.apply(syntactic_features)
        vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Semantic":
        X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
        return X_features, None

    elif phase == "Discourse":
        X_processed = X.apply(discourse_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Pragmatic":
        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
        return X_features, None
    
    return None, None


def evaluate_models(df: pd.DataFrame, selected_phase: str):
    """Trains and evaluates models using Stratified K-Fold Cross-Validation and SMOTE."""
    
    # 1. Data Preparation and Label Encoding
    X_raw = df['statement'].astype(str)
    y_raw = df['label']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    if len(np.unique(y)) < 2:
        st.error("Target column must contain at least two unique classes for classification.")
        return pd.DataFrame() 
    
    # 2. Feature Extraction (Apply to all data once per phase)
    # We only fit the vectorizer here to ensure it's consistent across all folds.
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame()
        
    # Convert X_features_full to a dense array if it's a DataFrame (for KFold indexing)
    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values
    
    # 3. K-Fold Setup
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    models_to_run = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    }

    model_metrics = {name: [] for name in models_to_run.keys()}

    for name, model in models_to_run.items():
        st.caption(f"🚀 Training {name} with {N_SPLITS}-Fold CV...")
        
        fold_metrics = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []
        }
        
        # We need the raw text for the feature application on each fold
        X_raw_list = X_raw.tolist()
        
        for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
            
            X_train_raw = pd.Series([X_raw_list[i] for i in train_index])
            X_test_raw = pd.Series([X_raw_list[i] for i in test_index])
            y_train = y[train_index]
            y_test = y[test_index]
            
            # Re-apply feature extraction to ensure proper handling of text data in the loop
            # If a vectorizer was used, apply transform using the fitted one.
            if vectorizer is not None:
                X_train = vectorizer.transform(X_train_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
                X_test = vectorizer.transform(X_test_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
            else:
                # Dense feature sets (Semantic/Pragmatic)
                X_train, _ = apply_feature_extraction(X_train_raw, selected_phase)
                X_test, _ = apply_feature_extraction(X_test_raw, selected_phase)
            
            
            # --- SMOTE PIPELINE ---
            # Create a pipeline to apply SMOTE *only* on the training data
            smote_pipeline = ImbPipeline([
                ('sampler', SMOTE(random_state=42, k_neighbors=3)),
                ('classifier', model)
            ])

            start_time = time.time()
            try:
                # MultinomialNB requires positive integers
                if name == "Naive Bayes":
                    X_train = X_train.abs().astype(int)
                    # Note: We skip SMOTE for MNB since it works poorly with synthetic samples
                    model.fit(X_train, y_train) 
                    clf = model
                else:
                    smote_pipeline.fit(X_train, y_train)
                    clf = smote_pipeline
                
                train_time = time.time() - start_time
                
                start_inference = time.time()
                y_pred = clf.predict(X_test)
                inference_time = (time.time() - start_inference) * 1000 
                
                # Metrics
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inference_time)

            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                # Append 0s for failed fold
                for key in fold_metrics: fold_metrics[key].append(0)
                continue

        # Calculate means across all folds
        if fold_metrics['accuracy']:
            model_metrics[name] = {
                "Model": name,
                "Accuracy": np.mean(fold_metrics['accuracy']) * 100,
                "F1-Score": np.mean(fold_metrics['f1']),
                "Precision": np.mean(fold_metrics['precision']),
                "Recall": np.mean(fold_metrics['recall']),
                "Training Time (s)": round(np.mean(fold_metrics['train_time']), 2),
                "Inference Latency (ms)": round(np.mean(fold_metrics['inference_time']), 2),
            }
        else:
             st.error(f"⚠️ {name} failed across all folds.")
             model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }

    results_list = list(model_metrics.values())
    return pd.DataFrame(results_list)

# ============================
# 4. HUMOR & CRITIQUE FUNCTIONS (REMAINS UNCHANGED)
# ============================

def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It’s concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It’s the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It’s the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It’s the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))


def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."

    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_model_row = df_results.loc[df_results['F1-Score'].idxmax()]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    
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
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                if not scraped_df.empty:
                    st.session_state['scraped_df'] = scraped_df
                    st.success(f"Scraping complete! {len(scraped_df)} claims successfully harvested.")
                else:
                    st.warning("Scraper returned no data. Try adjusting the date range or checking the target website.")
        
        st.divider()
        st.header("2. Analysis Configuration")
        
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
                with st.spinner(f"Engaging {selected_phase} features... training 4 models with {N_SPLITS}-Fold CV & SMOTE!"):
                    df_results = evaluate_models(st.session_state['scraped_df'], selected_phase)
                    st.session_state['df_results'] = df_results
                    st.session_state['selected_phase_run'] = selected_phase
                    st.success("Analysis complete! Prepare for the robust, cross-validated results.")


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

            st.dataframe(
                df_results[['Model', 'Accuracy', 'F1-Score', 'Training Time (s)', 'Inference Latency (ms)']],
                use_container_width=True,
                height=200
            )
            
            st.divider()
            st.subheader("Metric Comparison")
            
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)']
            plot_metric = st.selectbox("Metric to Plot:", metrics, index=1, key='plot_metric_center')
            
            df_plot = df_results[['Model', plot_metric]].set_index('Model')
            
            if 'Time' in plot_metric or 'Latency' in plot_metric:
                 st.bar_chart(df_plot, color="#FF5733")
            else:
                 st.bar_chart(df_plot, color="#33FF57")
            
            st.caption(f"Chart shows how each model performed on the selected metric using the **{st.session_state['selected_phase_run']}** features. Results are averaged over {N_SPLITS} folds.")


    # ============================
    # RIGHT COLUMN (Critique & Trade-offs)
    # ============================
    with col_right:
        st.header("4. Humorous Critique")
        
        if st.session_state['df_results'].empty:
            st.info("The models are currently in a coffee break. Results coming soon!")
        else:
            critique_text = generate_humorous_critique(st.session_state['df_results'], st.session_state['selected_phase_run'])
            st.markdown(critique_text)
            st.divider()
            
            st.subheader("Speed vs. Quality Trade-off")
            
            metrics_quality = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            metrics_speed = ['Training Time (s)', 'Inference Latency (ms)']
            
            x_axis = st.selectbox("X-Axis (Speed/Cost):", metrics_speed, key='x_axis', index=1)
            y_axis = st.selectbox("Y-Axis (Quality):", metrics_quality, key='y_axis', index=0)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            ax.scatter(df_results[x_axis], df_results[y_axis], s=150, alpha=0.7)
            
            for i, row in df_results.iterrows():
                ax.annotate(row['Model'], (row[x_axis] + 0.05 * df_results[x_axis].max(), row[y_axis] * 0.99), fontsize=9)
            
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Model Performance: {x_axis} vs. {y_axis}")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.caption("Look for models in the **bottom-left corner** for the best balance (Low Time, High Quality).")
            
# --- Run App ---
if __name__ == '__main__':
    N_SPLITS = 5 # Defined globally for use in app() and evaluate_models
    app()
