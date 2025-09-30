import json
import os
from datetime import datetime

# --- Configuration ---
# File paths
REPORTS_FILE = os.path.join("analysis", "analysis_reports.json")
README_FILE = "README.md"

# Markers in the README to define the section to be updated
START_MARKER = "<!-- START_ANALYSIS_SECTION -->"
END_MARKER = "<!-- END_ANALYSIS_SECTION -->"

# --- Main Logic ---

def get_rank_change_str(current_rank: int, prev_rank_map: dict, topic: str) -> str:
    """
    Generates a string representing the rank change of a topic.
    Returns 'NEW' for new topics, 'UP X' or 'DOWN X' for changes, and '-' for no change.
    """
    if topic not in prev_rank_map:
        return "NEW"
        
    prev_rank = prev_rank_map[topic]
    rank_change = prev_rank - current_rank
    
    if rank_change > 0:
        return f"UP {rank_change}"
    if rank_change < 0:
        return f"DOWN {abs(rank_change)}"
    return "-" # Represents no change in rank

def get_sentiment_change_str(current_score: float, prev_score: float) -> str:
    """
    Generates a string for sentiment change, using a small threshold to avoid noise.
    """
    threshold = 0.01
    if current_score > prev_score + threshold:
        return "(UP)"
    if current_score < prev_score - threshold:
        return "(DOWN)"
    return ""

def main():
    """
    Loads analysis data, generates a Markdown report, and injects it into the README.
    """
    print("Starting README update process.")
    
    # 1. Load the analysis reports JSON file
    try:
        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
            reports = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or parse '{REPORTS_FILE}'. Skipping README update.")
        return

    if not reports:
        print("Warning: No reports found in the analysis file. Skipping README update.")
        return

    # 2. Get the latest report and the one before it (if it exists)
    latest_report = reports[-1]
    previous_report = reports[-2] if len(reports) > 1 else None

    # 3. Prepare data for comparison
    # Create a map of topic -> rank for the previous day for efficient lookup
    prev_rank_map = {}
    if previous_report and "hot_topics" in previous_report:
        for i, topic_data in enumerate(previous_report["hot_topics"]):
            # Rank is index + 1
            prev_rank_map[topic_data["ngram"]] = i + 1

    # 4. Generate the new Markdown content for the report
    markdown_lines = []
    
    # Add a timestamp
    report_time = datetime.fromisoformat(latest_report["report_timestamp_utc"])
    markdown_lines.append(f"*Last Updated: {report_time.strftime('%Y-%m-%d %H:%M UTC')}*\n")

    # Generate Sentiment Analysis section
    sentiment = latest_report.get("sentiment_analysis", {})
    if sentiment:
        markdown_lines.append("### Sentiment Analysis")
        prev_score = previous_report.get("sentiment_analysis", {}).get("average_score", 0.0) if previous_report else 0.0
        sentiment_change = get_sentiment_change_str(sentiment['average_score'], prev_score)
        
        markdown_lines.append(f"- **Overall Mood Score**: `{sentiment['average_score']:.3f}` {sentiment_change}")
        markdown_lines.append(f"- **Distribution**: {sentiment['positive_percent']:.1f}% Positive, {sentiment['neutral_percent']:.1f}% Neutral, {sentiment['negative_percent']:.1f}% Negative")

    # Generate Hot Topics section
    hot_topics = latest_report.get("hot_topics", [])
    if hot_topics:
        markdown_lines.append("\n### Hot Topics")
        markdown_lines.append("| Rank | Change | Topic | Mentions |")
        markdown_lines.append("|:----:|:-------|:------|:--------:|")

        for i, topic_data in enumerate(hot_topics):
            rank = i + 1
            topic = topic_data["ngram"]
            mentions = topic_data["mentions"]
            change_str = get_rank_change_str(rank, prev_rank_map, topic)
            
            markdown_lines.append(f"| {rank} | {change_str} | `{topic}` | {mentions} |")

    # 5. Read the existing README, find the markers, and inject the new content
    try:
        with open(README_FILE, "r", encoding="utf-8") as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"Error: '{README_FILE}' not found. Cannot update.")
        return

    start_index = readme_content.find(START_MARKER)
    end_index = readme_content.find(END_MARKER)

    if start_index == -1 or end_index == -1:
        print(f"Error: Could not find the markers in '{README_FILE}'. Cannot update.")
        return

    # Build the new README content
    new_readme_content = (
        readme_content[:start_index + len(START_MARKER)] +
        "\n\n" +
        "\n".join(markdown_lines) +
        "\n\n" +
        readme_content[end_index:]
    )

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(new_readme_content)
    
    print("README.md has been successfully updated with the latest analysis.")

if __name__ == "__main__":
    main()
