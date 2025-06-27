import os
import json
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import requests
from bs4 import BeautifulSoup, Tag

# ==============================================================================
# 1. CENTRALIZED CONFIGURATION & PROFESSIONAL LOGGING SETUP
# ==============================================================================

class Config:
    """A single source of truth for all configuration."""
    SITE_URL: str = "https://status.cafe/"
    # MODIFICATION: Use a relative path. This will create a 'data' folder
    # in the same directory where the script is run.
    OUTPUT_DIR: str = "data"
    JSON_FILENAME: str = "statuses.json"
    LOG_FILENAME: str = "scraper.log"
    REQUEST_TIMEOUT_SECONDS: int = 15
    # MODIFICATION: Update this with your actual GitHub repo URL once you create it.
    REQUEST_HEADERS: Dict[str, str] = {
        'User-Agent': 'StatusCafeTrendScraper/1.0 (https://github.com/your-username/your-repo-name)'
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
# 2. ROBUST DATA HANDLING & FILE I/O
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
        # MODIFICATION: This check is now more robust for relative paths.
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
# 3. REFINED PARSING & SCRAPING LOGIC (No changes needed here)
# ==============================================================================

def parse_relative_time(time_str: str) -> Optional[datetime]:
    """
    Intelligently parses a relative time string into an absolute datetime object.
    Returns None if parsing fails.
    """
    now = datetime.now()
    normalized_str = time_str.lower().strip()
    
    if "now" in normalized_str or "just now" in normalized_str: return now
    if "yesterday" in normalized_str: return now - timedelta(days=1)

    normalized_str = re.sub(r'^(a|an)\s', '1 ', normalized_str)
    
    try:
        num_match = re.search(r'\d+', normalized_str)
        if not num_match:
            logging.warning(f"Could not find a number in time string: '{time_str}'")
            return None

        num = int(num_match.group(0))
        
        if 'minute' in normalized_str: return now - timedelta(minutes=num)
        if 'hour' in normalized_str: return now - timedelta(hours=num)
        if 'day' in normalized_str: return now - timedelta(days=num)
        if 'week' in normalized_str: return now - timedelta(weeks=num)
        if 'month' in normalized_str: return now - timedelta(days=num * 30)
        if 'year' in normalized_str: return now - timedelta(days=num * 365)
        
        logging.warning(f"Unrecognized time unit in string: '{time_str}'")
        return None
    except Exception as e:
        logging.error(f"Error parsing time string '{time_str}': {e}")
        return None

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

def parse_statuses_from_soup(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extracts all status data from a BeautifulSoup object defensively."""
    parsed_statuses = []
    status_elements = soup.select(Config.Selectors.STATUS_CONTAINER)
    
    if not status_elements:
        logging.warning(f"No status elements found with selector '{Config.Selectors.STATUS_CONTAINER}'. The website layout may have changed.")
        return []

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
            
            parsed_statuses.append({
                "id": status_id,
                "username": username,
                "text": text,
                "relative_time_on_site": relative_time,
            })
        except Exception as e:
            logging.error(f"An unexpected error occurred while parsing a single status: {e}")
            continue
    
    return parsed_statuses

# ==============================================================================
# 4. MAIN ORCHESTRATION LOGIC (No changes needed here)
# ==============================================================================

def main():
    """Main orchestration function for the scraper."""
    setup_logging()
    logging.info("--- Starting Status.Cafe Scraper Run ---")
    
    data_filepath = os.path.join(Config.OUTPUT_DIR, Config.JSON_FILENAME)
    existing_data = load_existing_data(data_filepath)
    logging.info(f"Loaded {len(existing_data)} existing statuses.")
    
    soup = fetch_page(Config.SITE_URL)
    if not soup:
        logging.critical("Could not fetch or parse the page. Aborting run.")
        return

    scraped_statuses = parse_statuses_from_soup(soup)
    if not scraped_statuses:
        logging.info("Scraping finished, but no statuses were parsed.")
        return
        
    new_statuses_added = 0
    for status in scraped_statuses:
        if status["id"] not in existing_data:
            absolute_time = parse_relative_time(status["relative_time_on_site"])
            
            existing_data[status["id"]] = {
                "username": status["username"],
                "text": status["text"],
                "timestamp_iso": absolute_time.isoformat() if absolute_time else None,
                "relative_time_on_site": status["relative_time_on_site"],
                "retrieval_date_iso": datetime.now().isoformat()
            }
            new_statuses_added += 1
    
    if new_statuses_added > 0:
        logging.info(f"Found {new_statuses_added} new statuses. Saving to file.")
        safe_save_data(data_filepath, existing_data)
    else:
        logging.info("No new statuses found on the page.")
        
    logging.info("--- Scraper run complete. ---")

if __name__ == "__main__":
    main()