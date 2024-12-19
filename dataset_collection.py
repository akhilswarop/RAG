import os
import argparse
import pandas as pd
import requests
from urllib.parse import urlparse
import time
import logging
from tqdm import tqdm

# ==============================
# Configuration and Defaults
# ==============================

# Default configuration values
DEFAULT_CSV_FOLDER = 'resume-csv'                # Folder containing CSV files
DEFAULT_DOWNLOAD_FOLDER = 'dataset/resumes'       # Folder to save downloaded resumes
DEFAULT_LINK_COLUMN = 'Resume URL'                # Column name in CSV files that contains the resume URLs
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
DEFAULT_RETRY_COUNT = 3                           # Number of retry attempts for failed downloads
DEFAULT_SLEEP_BETWEEN_DOWNLOADS = 1               # Seconds to wait between retries

# ==============================
# Logging Configuration
# ==============================

def setup_logging():
    """Configure logging for the script."""
    logger = logging.getLogger('ResumeDownloader')
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('download_resumes.log', mode='a')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logging()

# ==============================
# Utility Functions
# ==============================

def ensure_folder(folder_path):
    """
    Ensure that the specified folder exists. If not, create it.

    Args:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Created folder: {folder_path}")
    else:
        logger.info(f"Folder already exists: {folder_path}")

def get_all_csv_files(folder):
    """
    Retrieve all CSV file paths from the specified folder and its subfolders.

    Args:
        folder (str): Path to the folder containing CSV files.

    Returns:
        list: List of full paths to CSV files.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    if not csv_files:
        logger.warning(f"No CSV files found in folder: {folder}")
    else:
        logger.info(f"Found {len(csv_files)} CSV file(s) in folder: {folder}")
    return csv_files

def extract_links(csv_file, link_columns):
    """
    Extract download links from specified columns in the CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        link_columns (list): List of column names to extract links from.

    Returns:
        list: List of extracted URLs.
    """
    links = []
    try:
        df = pd.read_csv(csv_file)
        for column in link_columns:
            if column in df.columns:
                column_links = df[column].dropna().astype(str).tolist()
                links.extend(column_links)
                logger.info(f"Found {len(column_links)} link(s) in column '{column}' of {os.path.basename(csv_file)}.")
            else:
                logger.warning(f"Column '{column}' not found in {os.path.basename(csv_file)}. Skipping this column.")
        if not links:
            logger.warning(f"No links found in {csv_file}.")
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}")
    return links

def get_filename_from_url(url, session):
    """
    Extract the filename from the URL. If not present, generate one based on Content-Type.

    Args:
        url (str): The URL to extract the filename from.
        session (requests.Session): The session object to make HTTP requests.

    Returns:
        str: The extracted or generated filename.
    """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename or '.' not in filename:
        # If URL does not contain a filename or lacks an extension, generate one based on Content-Type
        try:
            head = session.head(url, allow_redirects=True, timeout=10)
            content_type = head.headers.get('Content-Type', '')
            if 'application/pdf' in content_type:
                extension = 'pdf'
            else:
                extension = 'pdf'  # Default to PDF since we're filtering for PDFs
            filename = f"resume_{int(time.time())}.{extension}"
        except Exception as e:
            logger.error(f"Failed to get Content-Type for {url}: {e}")
            filename = f"resume_{int(time.time())}.pdf"  # Default to PDF
    return filename

def download_file(url, folder, session, retry, sleep):
    """
    Download a PDF file from a URL to the specified folder.

    Args:
        url (str): The URL to download the file from.
        folder (str): The folder to save the downloaded file.
        session (requests.Session): The session object to make HTTP requests.
        retry (int): Number of retry attempts.
        sleep (int): Seconds to wait between retries.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:

        # Extract filename
        filename = get_filename_from_url(url, session)
        file_path = os.path.join(folder, filename)

        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"File already exists: {filename}. Skipping download.")
            return True

        for attempt in range(1, retry + 1):
            try:
                with session.get(url, headers={'User-Agent': DEFAULT_USER_AGENT}, stream=True, timeout=20) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors

                    # Validate Content-Type for PDF
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/pdf' not in content_type:
                        logger.info(f"Skipping non-PDF file with Content-Type: {content_type} - URL: {url}")
                        return False

                    # Download the file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"Downloaded: {filename}")
                    return True
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt} - Failed to download {url}: {e}")
                if attempt < retry:
                    logger.info(f"Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                else:
                    logger.error(f"Failed to download {url} after {retry} attempts.")
        return False
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return False

# ==============================
# Argument Parsing
# ==============================

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Sequentially download PDF resumes from CSV links.')
    parser.add_argument('--csv_folder', type=str, default=DEFAULT_CSV_FOLDER,
                        help='Folder containing CSV files (default: resume-csv)')
    parser.add_argument('--download_folder', type=str, default=DEFAULT_DOWNLOAD_FOLDER,
                        help='Folder to save downloaded resumes (default: dataset/resumes)')
    parser.add_argument('--link_columns', type=str, nargs='+', default=[DEFAULT_LINK_COLUMN],
                        help='Column name(s) with resume URLs (default: "Resume URL")')
    parser.add_argument('--retry', type=int, default=DEFAULT_RETRY_COUNT,
                        help='Number of retry attempts for failed downloads (default: 3)')
    parser.add_argument('--sleep', type=int, default=DEFAULT_SLEEP_BETWEEN_DOWNLOADS,
                        help='Seconds to wait between retries (default: 1)')
    return parser.parse_args()

# ==============================
# Main Function
# ==============================

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Log the start of the process
    logger.info("Resume Downloader Script Started.")
    logger.info(f"CSV Folder: {args.csv_folder}")
    logger.info(f"Download Folder: {args.download_folder}")
    logger.info(f"Link Columns: {args.link_columns}")
    logger.info(f"Retry Count: {args.retry}")
    logger.info(f"Sleep Between Retries: {args.sleep} seconds")

    # Ensure download folder exists
    ensure_folder(args.download_folder)

    # Get all CSV files
    csv_files = get_all_csv_files(args.csv_folder)
    if not csv_files:
        logger.error("No CSV files to process. Exiting.")
        return

    # Initialize a requests session
    with requests.Session() as session:
        # Iterate over each CSV file
        for csv_file in csv_files:
            logger.info(f"Processing CSV file: {os.path.basename(csv_file)}")
            links = extract_links(csv_file, args.link_columns)
            if links:
                # Use tqdm to display a progress bar
                for url in tqdm(links, desc=f"Downloading from {os.path.basename(csv_file)}", unit="resume"):
                    download_file(url, args.download_folder, session, args.retry, args.sleep)
            else:
                logger.warning(f"No valid links found in {csv_file}. Skipping download.")

    logger.info("Resume Downloader Script Completed.")

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()
