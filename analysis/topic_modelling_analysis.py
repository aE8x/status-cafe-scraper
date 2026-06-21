import os
import json
import logging
import re
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# ==============================================================================
# 1. CENTRALIZED CONFIGURATION
# ==============================================================================

class Config:
    """Configuration for the Trend and Sentiment Analyzer."""
    # --- File Paths ---
    DATA_DIR: str = "data"
    REPORT_DIR: str = "analysis"
    REPORT_JSON_FILENAME: str = "analysis_reports.json"
    LOG_FILENAME: str = "analysis/analyzer.log"
    
    # --- N-GRAM ANALYSIS WINDOWS ---
    CURRENT_ANALYSIS_HOURS: int = 336
    HISTORICAL_BASELINE_HOURS: int = 2160
    
    # --- N-Gram Trend Parameters ---
    N_GRAM_SIZES: List[int] = [1, 2, 3]
    MIN_MENTIONS_FOR_NGRAM_TREND: int = 2
    MIN_MENTIONS_FOR_NEW_TREND: int = 4
    NGRAM_SPIKE_MULTIPLIER: float = 2.5
    TOP_N_NGRAMS_TO_SHOW: int = 25

    # --- NLP Filtering Parameters ---
    SUB_NGRAM_OVERLAP_THRESHOLD: float = 0.80  # Filter out smaller phrase if 80%+ is in a larger phrase
    
    # Expanded stopwords targeting status.cafe culture, temporal words, and conversational filler
    CUSTOM_STOPWORDS: List[str] = [
        # 1. NLTK Gaps, Contractions, & Basic Verbs
        'im', 'ive', 'gonna', 'wan', 'na', 'ill', 'id', 'dont', 'cant', 'youre', 'theyre',
        'got', 'get', 'getting', 'gotta', 'make', 'making', 'made', 'say', 'said', 'saying', 
        'going', 'go', 'went', 'want', 'wants', 'wanted', 'need', 'needs', 'needed', 
        'think', 'thought', 'know', 'knew', 'feel', 'feels', 'feeling', 'look', 'looks', 
        'looking', 'put', 'putting', 'come', 'came', 'see', 'saw', 'seeing', 'try', 'trying',
        
        # 2. Status.cafe & Webmaster Meta
        'status', 'cafe', 'update', 'sitemap', 'site', 'website', 'button', 'layout', 
        'marquee', 'marquees', 'webmaster', 'domain', 'blog', 'post', 'posts', 'page', 
        'online', 'offline', 'link', 'url', 'http', 'https', 'www', 'com', 'org', 'net', 
        'neocities', 'html', 'css', 'web', 'internet', 'codeberg', 'wordpress', 'vgen',
        
        # 3. Temporal, Dates, & Countdowns
        'goodnight', 'morning', 'evening', 'night', 'day', 'days', 'today', 'tomorrow', 
        'yesterday', 'week', 'weeks', 'month', 'months', 'year', 'years', 'time', 'hour', 
        'hours', 'minute', 'minutes', 'till', 'ago', 'another', 'still', 'already', 'always', 
        'never', 'sometimes', 'soon', 'hourly', 'currently', 'recently',
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 
        'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        
        # 4. Conversational Fillers, Adverbs, & Expletives
        'like', 'man', 'one', 'ever', 'realsies', 'oooh', 'really', 'actually', 'fucking', 
        'fuck', 'shit', 'damn', 'god', 'bruh', 'bru', 'lmaoo', 'lmao', 'lmfao', 'omg', 
        'omggg', 'aaaaaa', 'yeah', 'yeahhh', 'urgh', 'ugh', 'much', 'many', 'every', 
        'everything', 'nothing', 'anything', 'someone', 'everyone', 'anyone', 'noone', 
        'way', 'least', 'whatever', 'whoever', 'whenever', 'however', 'good', 'bad', 
        'better', 'worse', 'best', 'worst', 'well', 'bit', 'lot', 'lots', 'stuff', 'things', 
        'thing', 'back', 'around', 'away', 'even', 'boy', 'girl', 'guy', 'dude',
        
        # 5. Multilingual Greetings & Common Bleed-over
        'buenos', 'noches', 'guten', 'nacht', 'bonne', 'nuit', 'meu', 'deus', 'bom', 'dia', 'ita'
    ]

    # Sentiment Analysis Parameters
    SENTIMENT_ANALYSIS_HOURS: int = 336

# ==============================================================================
# 2. LOGGING & INITIAL SETUP
# ==============================================================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(Config.LOG_FILENAME, encoding='utf-8', mode='w'), logging.StreamHandler()])

def setup_nltk():
    """Downloads necessary NLTK data, including the VADER lexicon."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logging.info("Downloading NLTK data (stopwords, vader_lexicon)...")
        nltk.download(['stopwords', 'vader_lexicon'], quiet=True)

# ==============================================================================
# 3. TREND & SENTIMENT ANALYZER
# ==============================================================================

class TrendSentimentAnalyzer:
    def __init__(self):
        self.report_filepath = os.path.join(Config.REPORT_DIR, Config.REPORT_JSON_FILENAME)
        self.df = None
        
        # Combine base English stopwords with our custom community list
        base_stopwords = set(stopwords.words('english'))
        self.stop_words = base_stopwords.union(Config.CUSTOM_STOPWORDS)
        
        self.sia = SentimentIntensityAnalyzer()

    def _discover_and_load_monthly_files(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        """
        Discovers and loads all monthly status files within the given date range.
        Returns a combined dictionary of all statuses from all months.
        """
        all_data = {}
        
        current_date = start_date.replace(day=1)
        end_month = end_date.replace(day=1)
        
        logging.info(f"Searching for monthly files from {current_date.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')}")
        
        while current_date <= end_month:
            year = current_date.year
            month = current_date.month
            
            year_dir = os.path.join(Config.DATA_DIR, str(year))
            filepath = os.path.join(year_dir, f"statuses_{year}_{month:02d}.json")
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        month_data = json.load(f)
                        all_data.update(month_data)
                        logging.info(f"Loaded {len(month_data)} statuses from {filepath}")
                except Exception as e:
                    logging.warning(f"Could not load {filepath}: {e}")
            else:
                logging.info(f"Monthly file not found: {filepath}")
            
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)
        
        return all_data

    def _load_and_prepare_data(self) -> bool:
        """
        Loads data from all available monthly files spanning the required time period.
        """
        logging.info(f"Loading data from monthly files in '{Config.DATA_DIR}'...")
        
        now = pd.Timestamp.now(tz='UTC')
        
        total_hours_needed = Config.CURRENT_ANALYSIS_HOURS + Config.HISTORICAL_BASELINE_HOURS
        earliest_date_needed = now - pd.Timedelta(hours=total_hours_needed)
        earliest_sentiment_date = now - pd.Timedelta(hours=Config.SENTIMENT_ANALYSIS_HOURS)
        
        start_date = min(earliest_date_needed, earliest_sentiment_date)
        
        logging.info(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
        
        all_data = self._discover_and_load_monthly_files(start_date, now)
        
        if not all_data:
            logging.critical("No data loaded from any monthly files. Cannot perform analysis.")
            return False
        
        logging.info(f"Total statuses loaded from all months: {len(all_data)}")
        
        self.df = pd.DataFrame.from_dict(all_data, orient='index')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp_iso'], errors='coerce')
        self.df.dropna(subset=['text', 'timestamp'], inplace=True)
        
        logging.info(f"DataFrame prepared with {len(self.df)} valid statuses")
        return True

    def _has_sufficient_data_span(self) -> bool:
        if self.df.empty:
            logging.warning("DataFrame is empty. Cannot perform analysis.")
            return False

        oldest_status_time = self.df['timestamp'].min()
        now = pd.Timestamp.now(tz='UTC')
        required_span_hours = Config.CURRENT_ANALYSIS_HOURS + Config.HISTORICAL_BASELINE_HOURS
        required_span = pd.Timedelta(hours=required_span_hours)
        actual_data_span = now - oldest_status_time

        if actual_data_span < required_span:
            required_days = required_span.days
            message = (
                f"\n--- ANALYSIS SKIPPED: INSUFFICIENT DATA ---\n"
                f"The oldest status is too recent for trend analysis (requires {required_days} days).\n"
            )
            print(message)
            logging.warning("Insufficient data for trend analysis.")
            return False
        
        logging.info("Data span check passed for trend analysis.")
        return True

    def _preprocess_text(self, text: str) -> List[str]:
        if not isinstance(text, str): return []
        
        # 1. Strip URLs (including raw domains and trailing paths like site.neocities.org/page)
        text_no_urls = re.sub(r'(?:http[s]?://\S+|www\.\S+|\b\S+\.(?:com|org|net|co|io|cafe)(?:/\S*)?)', '', text.lower())
        
        # 2. Tokenize into clean words (3+ characters, a-z only)
        raw_words = re.findall(r'\b[a-z]{3,}\b', text_no_urls)
        
        # 3. Filter using the combined stop words
        return [w for w in raw_words if w not in self.stop_words]

    # --- Part 1: N-Gram Trend Analysis ---
    def _run_ngram_analysis(self, df_current: pd.DataFrame, df_baseline: pd.DataFrame) -> List[dict]:
        logging.info("Starting N-Gram Trend Analysis...")
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

        # === Sub-ngram Redundancy Filter ===
        # Sort by word length (longest first), then count (highest first)
        spiking_ngrams.sort(key=lambda x: (len(x['ngram'].split()), x['count']), reverse=True)
        
        filtered_spikes = []
        
        for spike in spiking_ngrams:
            is_redundant = False
            
            # Pad with spaces to ensure accurate word-boundary matching
            padded_spike = f" {spike['ngram']} "
            
            for kept in filtered_spikes:
                padded_kept = f" {kept['ngram']} "
                
                # If the smaller string is perfectly nested inside the larger string
                if padded_spike in padded_kept:
                    # Calculate how much of the smaller phrase is accounted for by the larger phrase
                    overlap_ratio = kept['count'] / spike['count']
                    
                    if overlap_ratio >= Config.SUB_NGRAM_OVERLAP_THRESHOLD:
                        is_redundant = True
                        logging.debug(f"Filtered redundant n-gram: '{spike['ngram']}' (Absorbed by '{kept['ngram']}')")
                        break
            
            if not is_redundant:
                filtered_spikes.append(spike)

        return filtered_spikes

    def _get_ngrams_from_dataframe(self, df_period: pd.DataFrame) -> Dict[str, set]:
        ngrams_by_status_id = defaultdict(set)
        for status_id, row in df_period.iterrows():
            words = self._preprocess_text(row['text'])
            for n in Config.N_GRAM_SIZES:
                if len(words) >= n:
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i+n])
                        ngrams_by_status_id[ngram].add(status_id)
        return ngrams_by_status_id

    # Part 2: Sentiment Analysis
    def _run_sentiment_analysis(self, df_sentiment_window: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes the sentiment of statuses within a given timeframe."""
        logging.info(f"Starting Sentiment Analysis for the last {Config.SENTIMENT_ANALYSIS_HOURS} hours...")
        if df_sentiment_window.empty:
            logging.warning("No statuses found in the sentiment analysis window.")
            return {}

        # Calculate sentiment for each status
        df_sentiment_window['sentiment'] = df_sentiment_window['text'].apply(
            lambda text: self.sia.polarity_scores(text)['compound']
        )

        # Aggregate results
        avg_sentiment = df_sentiment_window['sentiment'].mean()
        total_statuses = len(df_sentiment_window)
        
        # Categorize statuses
        positive_count = df_sentiment_window[df_sentiment_window['sentiment'] > 0.05].shape[0]
        negative_count = df_sentiment_window[df_sentiment_window['sentiment'] < -0.05].shape[0]
        neutral_count = total_statuses - positive_count - negative_count
        
        # Find most positive and negative examples
        most_positive_status = df_sentiment_window.loc[df_sentiment_window['sentiment'].idxmax()]
        most_negative_status = df_sentiment_window.loc[df_sentiment_window['sentiment'].idxmin()]

        return {
            "average_score": avg_sentiment,
            "positive_percent": (positive_count / total_statuses) * 100 if total_statuses > 0 else 0,
            "negative_percent": (negative_count / total_statuses) * 100 if total_statuses > 0 else 0,
            "neutral_percent": (neutral_count / total_statuses) * 100 if total_statuses > 0 else 0,
            "most_positive_example": {
                "text": most_positive_status['text'],
                "user": most_positive_status['username'],
                "score": most_positive_status['sentiment']
            },
            "most_negative_example": {
                "text": most_negative_status['text'],
                "user": most_negative_status['username'],
                "score": most_negative_status['sentiment']
            }
        }

    # --- Part 3: Reporting ---
    def _save_report_to_json(self, report_data: Dict[str, Any]):
        logging.info(f"Saving report to '{self.report_filepath}'...")
        report_dir = os.path.dirname(self.report_filepath)
        if not os.path.exists(report_dir): os.makedirs(report_dir)
        all_reports = []
        try:
            if os.path.exists(self.report_filepath):
                with open(self.report_filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content: all_reports = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            all_reports = []
        all_reports.append(report_data)
        temp_filepath = self.report_filepath + ".tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_reports, f, indent=4, ensure_ascii=False)
            os.replace(temp_filepath, self.report_filepath)
            logging.info(f"Successfully saved {len(all_reports)} reports to '{self.report_filepath}'")
        except IOError as e:
            logging.critical(f"Could not save report data. Error: {e}")
        finally:
            if os.path.exists(temp_filepath): os.remove(temp_filepath)

    def _generate_report(self, spiking_ngrams: List[dict], sentiment_results: Dict):
        now = pd.Timestamp.now(tz='UTC')
        
        report_data_structured = { 
            "report_timestamp_utc": now.isoformat(), 
            "hot_topics": [], 
            "sentiment_analysis": sentiment_results,
            "summary_message": None 
        }

        print(f"\n--- Analysis Report ---")
        print(f"Analysis complete at: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        
        print(f"\n--- HOT TOPICS (Last {Config.CURRENT_ANALYSIS_HOURS // 24} Days vs. Past {Config.HISTORICAL_BASELINE_HOURS // 24} Days) ---")
        if not spiking_ngrams:
            no_trends_msg = "No phrases met the trending threshold."
            print(no_trends_msg)
            report_data_structured["summary_message"] = no_trends_msg
        else:
            # Sort by change (descending) then by count (descending) to resolve 'inf' ties fairly
            sorted_ngrams = sorted(spiking_ngrams, key=lambda x: (x['change'], x['count']), reverse=True)[:Config.TOP_N_NGRAMS_TO_SHOW]
            for item in sorted_ngrams:
                change_str = "NEW" if item['change'] == float('inf') else f"{item['change']:.1f}x"
                print(f"- \"{item['ngram']}\" ({item['count']} mentions, {change_str} increase)")
                report_data_structured["hot_topics"].append({ "ngram": item['ngram'], "mentions": item['count'], "change": change_str })

        print(f"\n--- SENTIMENT (Last {Config.SENTIMENT_ANALYSIS_HOURS} Hours) ---")
        if not sentiment_results:
            print("Not enough data to perform sentiment analysis for the period.")
        else:
            print(f"Overall Mood: {sentiment_results['average_score']:.2f} (from -1 to 1)")
            print(f"Distribution: {sentiment_results['positive_percent']:.1f}% Positive | {sentiment_results['neutral_percent']:.1f}% Neutral | {sentiment_results['negative_percent']:.1f}% Negative")
            print("-" * 20)
            print(f"Most Positive: \"{sentiment_results['most_positive_example']['text']}\" (by {sentiment_results['most_positive_example']['user']})")
            print(f"Most Negative: \"{sentiment_results['most_negative_example']['text']}\" (by {sentiment_results['most_negative_example']['user']})")
        
        print("-" * 35)
        self._save_report_to_json(report_data_structured)

    def run_analysis(self):
        if not self._load_and_prepare_data(): return

        spiking_ngrams = []
        if self._has_sufficient_data_span():
            now = pd.Timestamp.now(tz='UTC')
            current_boundary = now - pd.Timedelta(hours=Config.CURRENT_ANALYSIS_HOURS)
            baseline_boundary = current_boundary - pd.Timedelta(hours=Config.HISTORICAL_BASELINE_HOURS)
            df_current = self.df[self.df['timestamp'] >= current_boundary].copy()
            df_baseline = self.df[(self.df['timestamp'] >= baseline_boundary) & (self.df['timestamp'] < current_boundary)].copy()
            logging.info(f"Trend analysis windows: {len(df_current)} current, {len(df_baseline)} baseline.")
            spiking_ngrams = self._run_ngram_analysis(df_current, df_baseline)
        
        now = pd.Timestamp.now(tz='UTC')
        sentiment_boundary = now - pd.Timedelta(hours=Config.SENTIMENT_ANALYSIS_HOURS)
        df_sentiment_window = self.df[self.df['timestamp'] >= sentiment_boundary].copy()
        sentiment_results = self._run_sentiment_analysis(df_sentiment_window)

        self._generate_report(spiking_ngrams, sentiment_results)

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    setup_logging()
    setup_nltk()
    logging.info("--- Starting Trend & Sentiment Analyzer Run ---")
    analyzer = TrendSentimentAnalyzer()
    analyzer.run_analysis()
    logging.info("--- Analyzer run complete. ---")

if __name__ == "__main__":
    main()
