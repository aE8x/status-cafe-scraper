import os
import json
import logging
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# NLTK is used for more advanced text processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ==============================================================================
# 1. CENTRALIZED CONFIGURATION
# ==============================================================================

class Config:
    """Configuration for the Hybrid Analyzer."""
    # --- File Paths ---
    # MODIFICATION: Path to the scraper's output folder.
    # It's one level up from 'analysis' and then into 'data'.
    # Since the script is run from the repo root, the path is simply "data".
    DATA_DIR: str = "data"
    STATUS_JSON_FILENAME: str = "statuses.json"
    
    # MODIFICATION: Define the output directory and report filename.
    # This will save the report inside the 'analysis' folder.
    REPORT_DIR: str = "analysis"
    REPORT_JSON_FILENAME: str = "analysis_reports.json"

    # Log file will also be saved in the analysis directory.
    LOG_FILENAME: str = "analysis/analyzer.log"
    
    # --- N-GRAM ANALYSIS WINDOWS ---
    CURRENT_ANALYSIS_HOURS: int = 336      # We analyze the last 14 days for hot topics.
    HISTORICAL_BASELINE_HOURS: int = 4320   # We compare against the 180 days prior to that.
    
    # --- N-Gram Trend Parameters (Microscope) ---
    N_GRAM_SIZES: List[int] = [1, 2, 3]
    # **** NEW DUAL THRESHOLD LOGIC ****
    MIN_MENTIONS_FOR_NGRAM_TREND: int = 2 # For existing topics that are spiking.
    MIN_MENTIONS_FOR_NEW_TREND: int = 4   # Stricter threshold for brand-new topics.
    NGRAM_SPIKE_MULTIPLIER: float = 2.5
    TOP_N_NGRAMS_TO_SHOW: int = 15

    # --- Topic Modeling Parameters (Telescope) ---
    TOPIC_ANALYSIS_HOURS: int = 168
    NUM_TOPICS: int = 15
    TOP_WORDS_PER_TOPIC: int = 6
    NUM_TOP_ACTIVE_TOPICS_TO_SHOW: int = 5
    VECTORIZER_MAX_DF: float = 0.90
    VECTORIZER_MIN_DF: int = 5

# ==============================================================================
# 2. LOGGING & INITIAL SETUP
# ==============================================================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(Config.LOG_FILENAME, encoding='utf-8', mode='w'), logging.StreamHandler()])

def setup_nltk():
    """Now also downloads the Part-of-Speech tagger."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        logging.info("Downloading NLTK data (stopwords, wordnet, averaged_perceptron_tagger)...")
        nltk.download(['stopwords', 'wordnet', 'averaged_perceptron_tagger'], quiet=True)

# ==============================================================================
# 3. HYBRID ANALYZER
# ==============================================================================

class HybridAnalyzer:
    def __init__(self):
        self.status_filepath = os.path.join(Config.DATA_DIR, Config.STATUS_JSON_FILENAME)
        self.report_filepath = os.path.join(Config.REPORT_DIR, Config.REPORT_JSON_FILENAME)
        self.df = None
        self.stop_words = set(stopwords.words('english')).union(['im', 'ive', 'gonna', 'wan', 'na'])
        self.lemmatizer = WordNetLemmatizer()

    def _load_and_prepare_data(self) -> bool:
        logging.info(f"Loading data from '{self.status_filepath}'...")
        try:
            with open(self.status_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.critical(f"Could not load or parse status file: {e}"); return False
        
        self.df = pd.DataFrame.from_dict(data, orient='index')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp_iso'], errors='coerce')
        self.df.dropna(subset=['text', 'timestamp'], inplace=True)
        return True

    # NEW METHOD: Gatekeeper function to check data history.
    def _has_sufficient_data_span(self) -> bool:
        """
        Checks if the dataset contains enough historical data for a meaningful comparison.
        The oldest status must be older than the combined current and baseline analysis windows.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty. Cannot perform analysis.")
            return False

        oldest_status_time = self.df['timestamp'].min()
        now = pd.Timestamp.now(tz='UTC')

        # The total time span required for a valid analysis (baseline + current)
        required_span_hours = Config.CURRENT_ANALYSIS_HOURS + Config.HISTORICAL_BASELINE_HOURS
        required_span = pd.Timedelta(hours=required_span_hours)
        
        # The actual time span of data available in the dataset
        actual_data_span = now - oldest_status_time

        if actual_data_span < required_span:
            required_days = required_span.days
            message = (
                f"\n--- ANALYSIS SKIPPED: INSUFFICIENT DATA ---\n"
                f"The oldest status in the dataset is from {oldest_status_time.strftime('%Y-%m-%d')}.\n"
                f"A complete analysis requires at least {required_days} days of historical data "
                f"(for a {Config.HISTORICAL_BASELINE_HOURS // 24}-day baseline and "
                f"{Config.CURRENT_ANALYSIS_HOURS // 24}-day current window).\n"
                f"Please gather more data before running the analysis again.\n"
                f"---------------------------------------------"
            )
            print(message)
            logging.warning("Insufficient data for analysis. The oldest status is too recent.")
            return False
        
        logging.info("Data span check passed. Sufficient historical data available.")
        return True

    def _preprocess_text(self, text: str, for_lda: bool = False) -> List[str]:
        if not isinstance(text, str): return []
        
        if not for_lda:
            raw_words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return [w for w in raw_words if w not in self.stop_words]

        words = nltk.word_tokenize(text.lower())
        pos_tagged_words = nltk.pos_tag(words)
        
        meaningful_words = []
        for word, tag in pos_tagged_words:
            if tag.startswith('NN') or tag.startswith('JJ'):
                if word.isalpha() and len(word) >= 3 and word not in self.stop_words:
                    meaningful_words.append(self.lemmatizer.lemmatize(word))
        return meaningful_words

    # --- Part 1: N-Gram Trend Analysis ---
    def _run_ngram_analysis(self, df_current: pd.DataFrame, df_baseline: pd.DataFrame) -> List[dict]:
        logging.info("Starting Part 1: N-Gram Trend Analysis...")
        
        baseline_ngrams = self._get_ngrams_from_dataframe(df_baseline)
        current_ngrams = self._get_ngrams_from_dataframe(df_current)

        baseline_freq = {ng: len(ids) / len(df_baseline) * 1000 for ng, ids in baseline_ngrams.items()} if len(df_baseline) > 0 else {}
        current_freq = {ng: len(ids) / len(df_current) * 1000 for ng, ids in current_ngrams.items()} if len(df_current) > 0 else {}

        spiking_ngrams = []
        for ngram, freq in current_freq.items():
            count = len(current_ngrams[ngram])
            base_freq = baseline_freq.get(ngram, 0)
            is_new = (base_freq == 0)

            if is_new:
                if count < Config.MIN_MENTIONS_FOR_NEW_TREND: continue
            else:
                if count < Config.MIN_MENTIONS_FOR_NGRAM_TREND: continue
            
            change = float('inf') if is_new else freq / base_freq

            if change >= Config.NGRAM_SPIKE_MULTIPLIER:
                spiking_ngrams.append({'ngram': ngram, 'count': count, 'change': change})
        
        return spiking_ngrams

    def _get_ngrams_from_dataframe(self, df_period: pd.DataFrame) -> Dict[str, set]:
        ngrams_by_status_id = defaultdict(set)
        for status_id, row in df_period.iterrows():
            words = self._preprocess_text(row['text'], for_lda=False)
            for n in Config.N_GRAM_SIZES:
                if len(words) >= n:
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i+n])
                        ngrams_by_status_id[ngram].add(status_id)
        return ngrams_by_status_id

    # --- Part 2: Broad Topic Analysis ---
    def _run_topic_analysis(self, df_topic_window: pd.DataFrame) -> Dict[int, str]:
        logging.info("Starting Part 2: Broad Topic Analysis with POS Tagging...")
        df_topic_window['processed_lda'] = df_topic_window['text'].apply(lambda x: " ".join(self._preprocess_text(x, for_lda=True)))
        
        vectorizer = TfidfVectorizer(max_df=Config.VECTORIZER_MAX_DF, min_df=Config.VECTORIZER_MIN_DF)
        tfidf_matrix = vectorizer.fit_transform(df_topic_window['processed_lda'])
        
        lda = LatentDirichletAllocation(n_components=Config.NUM_TOPICS, random_state=42)
        lda.fit(tfidf_matrix)
        
        topic_words_map = {i: ", ".join([vectorizer.get_feature_names_out()[t] for t in topic.argsort()[:-Config.TOP_WORDS_PER_TOPIC - 1:-1]]) for i, topic in enumerate(lda.components_)}
        return topic_words_map

    # --- Part 3: Reporting ---
    def _save_report_to_json(self, report_data: Dict[str, Any]):
        """
        Safely loads existing reports, appends the new one, and saves the list
        back to a JSON file using an atomic write operation.
        """
        logging.info(f"Saving report to '{self.report_filepath}'...")
        
        # Ensure the output directory exists
        report_dir = os.path.dirname(self.report_filepath)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        # --- Step 1: Load existing data safely ---
        all_reports = []
        try:
            if os.path.exists(self.report_filepath):
                with open(self.report_filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Handle empty file case
                    if content:
                        all_reports = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Could not read/parse existing report file '{self.report_filepath}'. A new file will be created. Error: {e}")
            all_reports = []
        
        # --- Step 2: Append new data (This part was correct) ---
        all_reports.append(report_data)

        # --- Step 3: Save data back to file using an atomic write (This part was missing/can be improved) ---
        temp_filepath = self.report_filepath + ".tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                # The list of reports is the top-level object in the JSON
                json.dump(all_reports, f, indent=4, ensure_ascii=False)
            
            # Atomically replace the old file with the new one
            os.replace(temp_filepath, self.report_filepath)
            logging.info(f"Successfully saved {len(all_reports)} reports to '{self.report_filepath}'")

        except IOError as e:
            logging.critical(f"Could not save report data to '{self.report_filepath}'. Error: {e}")
        finally:
            # Clean up the temp file if it still exists
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    def _generate_report(self, spiking_ngrams: List[dict], topic_words_map: Dict):
        now = pd.Timestamp.now(tz='UTC')
        now_str = now.strftime('%Y-%m-%d %H:%M UTC')
        
        report_data_structured = { "report_timestamp_utc": now.isoformat(), "hot_topics": [], "broad_topics": [], "summary_message": None }

        print(f"\n--- Hybrid Analysis Report ---")
        print(f"Analysis complete at: {now_str}")
        
        print(f"\n--- HOT TOPICS (Last {Config.CURRENT_ANALYSIS_HOURS // 24} Days vs. Past {Config.HISTORICAL_BASELINE_HOURS // 24} Days) ---")
        if not spiking_ngrams:
            no_trends_msg = f"No phrases met the trending threshold (New: >={Config.MIN_MENTIONS_FOR_NEW_TREND}, Spiking: >={Config.MIN_MENTIONS_FOR_NGRAM_TREND} mentions & >{Config.NGRAM_SPIKE_MULTIPLIER}x freq. increase)."
            print(no_trends_msg)
            report_data_structured["summary_message"] = no_trends_msg
        else:
            sorted_ngrams = sorted(spiking_ngrams, key=lambda x: x['change'], reverse=True)[:Config.TOP_N_NGRAMS_TO_SHOW]
            for item in sorted_ngrams:
                change_str = "NEW" if item['change'] == float('inf') else f"{item['change']:.1f}x"
                print(f"- \"{item['ngram']}\" ({item['count']} mentions, {change_str} increase)")
                report_data_structured["hot_topics"].append({ "ngram": item['ngram'], "mentions": item['count'], "change": change_str })

        print(f"\n--- BROAD TOPICS (Last {Config.TOPIC_ANALYSIS_HOURS // 24} Days) ---")
        if topic_words_map:
            for topic_id, words_str in list(topic_words_map.items())[:Config.NUM_TOP_ACTIVE_TOPICS_TO_SHOW]:
                first_noun = words_str.split(', ')[0].capitalize()
                print(f"Topic: {first_noun} ({words_str})")
                report_data_structured["broad_topics"].append({ "topic_name": first_noun, "keywords": words_str })
        
        print("-" * 35)

        self._save_report_to_json(report_data_structured)

    # MODIFIED: The main orchestrator now includes the data span check.
    def run_analysis(self):
        if not self._load_and_prepare_data(): return

        # NEW: Check if there's enough data before proceeding.
        if not self._has_sufficient_data_span():
            return # Stop execution if data span is insufficient.

        now = pd.Timestamp.now(tz='UTC')
        current_boundary = now - pd.Timedelta(hours=Config.CURRENT_ANALYSIS_HOURS)
        baseline_boundary = current_boundary - pd.Timedelta(hours=Config.HISTORICAL_BASELINE_HOURS)
        topic_window_boundary = now - pd.Timedelta(hours=Config.TOPIC_ANALYSIS_HOURS)

        df_current = self.df[self.df['timestamp'] >= current_boundary].copy()
        df_baseline = self.df[(self.df['timestamp'] >= baseline_boundary) & (self.df['timestamp'] < current_boundary)].copy()
        df_topic_window = self.df[self.df['timestamp'] >= topic_window_boundary].copy()
        
        logging.info(f"Current analysis window has {len(df_current)} statuses. Historical baseline has {len(df_baseline)} statuses.")

        spiking_ngrams = self._run_ngram_analysis(df_current, df_baseline)
        topic_words_map = self._run_topic_analysis(df_topic_window)

        self._generate_report(spiking_ngrams, topic_words_map)

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    setup_logging()
    setup_nltk()
    logging.info("--- Starting Hybrid Analyzer Run ---")
    analyzer = HybridAnalyzer()
    analyzer.run_analysis()
    logging.info("--- Analyzer run complete. ---")

if __name__ == "__main__":
    main()
