import os
import json
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

import requests
from bs4 import BeautifulSoup, Tag

# ==============================================================================
# 1. CENTRALIZED CONFIGURATION & PROFESSIONAL LOGGING SETUP
# ==============================================================================

class Config:
    """A single source of truth for all configuration."""
    SITE_URL: str = "https://status.cafe/"
    OUTPUT_DIR: str = "data"
    LOG_FILENAME: str = "scraper.log"
    REQUEST_TIMEOUT_SECONDS: int = 15
    REQUEST_HEADERS: Dict[str, str] = {
        'User-Agent': 'StatusCafeTrendScraper/1.0 (https://github.com/aE8x/status-cafe-scraper)'
    }

    # Selectors are centralized for easy updates if the site changes
    class Selectors:
        STATUS_CONTAINER: str = "article.status"
        USERNAME_DIV: str = "div.status-username"
        USERNAME_LINK: str = "div.status-username a"
        STATUS_TEXT: str = "p.status-content"

def setup_logging():
    """Configures logging to file and console for robust error tracking."""
    log_dir = os.path.dirname(Config.LOG_FILENAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILENAME, encoding='utf-8'),
            logging.StreamHandler() # Also print to console (IMPORTANT for GitHub Actions logs)
        ]
    )

# ==============================================================================
# 2. DYNAMIC FILE PATH MANAGEMENT
# ==============================================================================

def get_current_data_filepath() -> str:
    """
    Returns the filepath for the current month's status file.
    Creates year directory if it doesn't exist.
    Format: data/YYYY/statuses_YYYY_MM.json
    """
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    
    year_dir = os.path.join(Config.OUTPUT_DIR, str(year))
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
        logging.info(f"Created year directory: {year_dir}")
    
    filename = f"statuses_{year}_{month:02d}.json"
    return os.path.join(year_dir, filename)

def get_excluded_data_filepath() -> str:
    """
    Returns the filepath for the current month's excluded status file.
    Creates excluded_statuses directory if it doesn't exist.
    Format: data/excluded_statuses/excluded_YYYY_MM.json
    """
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    
    excluded_dir = os.path.join(Config.OUTPUT_DIR, "excluded_statuses")
    if not os.path.exists(excluded_dir):
        os.makedirs(excluded_dir)
        logging.info(f"Created excluded statuses directory: {excluded_dir}")
    
    filename = f"excluded_{year}_{month:02d}.json"
    return os.path.join(excluded_dir, filename)

# ==============================================================================
# 3. ROBUST DATA HANDLING & FILE I/O
# ==============================================================================

def load_existing_data(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Safely loads data from the JSON file, handling corruption."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Data file '{filepath}' is corrupted. Starting fresh, but check the file!")
        return {}

def safe_save_data(filepath: str, data: Dict[str, Dict[str, Any]]):
    """
    Saves data atomically to prevent corruption.
    Writes to a temporary file first, then renames it to the final destination.
    """
    temp_filepath = filepath + ".tmp"
    final_dir = os.path.dirname(filepath)

    try:
        if final_dir and not os.path.exists(final_dir):
            os.makedirs(final_dir)
        
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        os.replace(temp_filepath, filepath)
        logging.info(f"Successfully saved {len(data)} statuses to '{filepath}'")

    except IOError as e:
        logging.critical(f"Could not save data to '{filepath}'. Error: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during file save: {e}")
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# ==============================================================================
# 4. REFINED PARSING & SCRAPING LOGIC WITH AGE FILTERING
# ==============================================================================

def parse_relative_time(time_str: str) -> Tuple[Optional[datetime], bool]:
    """
    Intelligently parses a relative time string into an absolute datetime object.
    Returns (datetime, is_acceptable) tuple.
    - is_acceptable is True if the status is less than 24 hours old
    - is_acceptable is False if the status is 1 day or older
    Returns (None, False) if parsing fails.
    """
    now = datetime.now(timezone.utc)
    normalized_str = time_str.lower().strip()
    
    # Immediate acceptance cases (definitely < 24 hours)
    if "now" in normalized_str or "just now" in normalized_str:
        return now, True
    
    # Yesterday means >= 24 hours, so reject
    if "yesterday" in normalized_str:
        return now - timedelta(days=1), False

    # Replace "a/an" with "1"
    normalized_str = re.sub(r'^(a|an)\s', '1 ', normalized_str)
    
    try:
        num_match = re.search(r'\d+', normalized_str)
        if not num_match:
            logging.warning(f"Could not find a number in time string: '{time_str}'")
            return None, False

        num = int(num_match.group(0))
        
        # Check for time units and determine acceptability
        if 'second' in normalized_str:
            return now - timedelta(seconds=num), True
        if 'minute' in normalized_str:
            return now - timedelta(minutes=num), True
        if 'hour' in normalized_str:
            # Accept only if less than 24 hours
            if num < 24:
                return now - timedelta(hours=num), True
            else:
                return now - timedelta(hours=num), False
        
        # Day, week, month, year are all >= 24 hours, so reject
        if 'day' in normalized_str:
            return now - timedelta(days=num), False
        if 'week' in normalized_str:
            return now - timedelta(weeks=num), False
        if 'month' in normalized_str:
            return now - timedelta(days=num * 30), False
        if 'year' in normalized_str:
            return now - timedelta(days=num * 365), False
        
        logging.warning(f"Unrecognized time unit in string: '{time_str}'")
        return None, False
    except Exception as e:
        logging.error(f"Error parsing time string '{time_str}': {e}")
        return None, False

def fetch_page(url: str) -> Optional[BeautifulSoup]:
    """Fetches and parses a web page, handling network errors."""
    try:
        response = requests.get(
            url,
            headers=Config.REQUEST_HEADERS,
            timeout=Config.REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch page '{url}'. Reason: {e}")
        return None

def parse_statuses_from_soup(soup: BeautifulSoup) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts all status data from a BeautifulSoup object defensively.
    Returns two lists: (acceptable_statuses, excluded_statuses)
    - acceptable_statuses: statuses less than 24 hours old
    - excluded_statuses: statuses 24 hours or older
    """
    acceptable_statuses = []
    excluded_statuses = []
    status_elements = soup.select(Config.Selectors.STATUS_CONTAINER)
    
    if not status_elements:
        logging.warning(f"No status elements found with selector '{Config.Selectors.STATUS_CONTAINER}'. The website layout may have changed.")
        return [], []

    for status_element in status_elements:
        try:
            username_div = status_element.select_one(Config.Selectors.USERNAME_DIV)
            username_link = status_element.select_one(Config.Selectors.USERNAME_LINK)
            text_elem = status_element.select_one(Config.Selectors.STATUS_TEXT)

            if not all([username_div, username_link, text_elem]):
                logging.warning("Skipping a malformed status element (missing user, text, or time div).")
                continue

            username = username_link.get_text(strip=True)
            text = text_elem.get_text(strip=True)
            
            full_header_text = username_div.get_text(strip=True)
            relative_time = full_header_text.replace(username, '', 1).strip()
            
            status_id = hashlib.sha256(f"{username}:{text}".encode('utf-8')).hexdigest()
            
            status_dict = {
                "id": status_id,
                "username": username,
                "text": text,
                "relative_time_on_site": relative_time,
            }
            
            # Parse time and check acceptability
            absolute_time, is_acceptable = parse_relative_time(relative_time)
            
            if is_acceptable:
                acceptable_statuses.append(status_dict)
            else:
                excluded_statuses.append(status_dict)
                
        except Exception as e:
            logging.error(f"An unexpected error occurred while parsing a single status: {e}")
            continue
    
    return acceptable_statuses, excluded_statuses

# ==============================================================================
# 5. MAIN ORCHESTRATION LOGIC
# ==============================================================================

def main():
    """Main orchestration function for the scraper."""
    setup_logging()
    logging.info("--- Starting Status.Cafe Scraper Run ---")
    
    # Get current month's file paths
    data_filepath = get_current_data_filepath()
    excluded_filepath = get_excluded_data_filepath()
    
    logging.info(f"Target data file: {data_filepath}")
    logging.info(f"Target excluded file: {excluded_filepath}")
    
    # Load existing data for both files
    existing_data = load_existing_data(data_filepath)
    existing_excluded = load_existing_data(excluded_filepath)
    
    logging.info(f"Loaded {len(existing_data)} existing statuses from current month.")
    logging.info(f"Loaded {len(existing_excluded)} existing excluded statuses from current month.")
    
    soup = fetch_page(Config.SITE_URL)
    if not soup:
        logging.critical("Could not fetch or parse the page. Aborting run.")
        return

    acceptable_statuses, excluded_statuses = parse_statuses_from_soup(soup)
    
    if not acceptable_statuses and not excluded_statuses:
        logging.info("Scraping finished, but no statuses were parsed.")
        return
    
    logging.info(f"Scraped {len(acceptable_statuses)} acceptable statuses (< 24 hours old)")
    logging.info(f"Scraped {len(excluded_statuses)} excluded statuses (>= 24 hours old)")
    
    # Process acceptable statuses
    new_statuses_added = 0
    for status in acceptable_statuses:
        if status["id"] not in existing_data:
            absolute_time, _ = parse_relative_time(status["relative_time_on_site"])
            
            existing_data[status["id"]] = {
                "username": status["username"],
                "text": status["text"],
                "timestamp_iso": absolute_time.isoformat() if absolute_time else None,
                "relative_time_on_site": status["relative_time_on_site"],
                "retrieval_date_iso": datetime.now(timezone.utc).isoformat()
            }
            new_statuses_added += 1
    
    # Process excluded statuses
    new_excluded_added = 0
    for status in excluded_statuses:
        if status["id"] not in existing_excluded:
            absolute_time, _ = parse_relative_time(status["relative_time_on_site"])
            
            existing_excluded[status["id"]] = {
                "username": status["username"],
                "text": status["text"],
                "timestamp_iso": absolute_time.isoformat() if absolute_time else None,
                "relative_time_on_site": status["relative_time_on_site"],
                "retrieval_date_iso": datetime.now(timezone.utc).isoformat()
            }
            new_excluded_added += 1
    
    # Save both files if there are changes
    if new_statuses_added > 0:
        logging.info(f"Found {new_statuses_added} new acceptable statuses. Saving to file.")
        safe_save_data(data_filepath, existing_data)
    else:
        logging.info("No new acceptable statuses found on the page.")
    
    if new_excluded_added > 0:
        logging.info(f"Found {new_excluded_added} new excluded statuses. Saving to excluded file.")
        safe_save_data(excluded_filepath, existing_excluded)
    else:
        logging.info("No new excluded statuses found on the page.")
        
    logging.info("--- Scraper run complete. ---")

if __name__ == "__main__":
    main()
