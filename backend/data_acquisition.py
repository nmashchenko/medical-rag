import feedparser
from datetime import datetime
import requests
import os
import shutil

def clear_folder(folder: str):
    """
    Clear all files and subdirectories from the specified folder.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder, exist_ok=True)

def fetch_arxiv_articles(title: str, start_date: datetime, end_date: datetime, max_results: int = 200):
    """
    Fetch articles from arXiv based on a title search and a submitted date range.

    Parameters:
        title (str): The term to search for in the title (e.g., "medical")
        start_date (datetime): Start of the date range (in GMT)
        end_date (datetime): End of the date range (in GMT)
        max_results (int): Maximum number of results to retrieve from the API

    Returns:
        List of feedparser entries.
    """
    # Format dates as YYYYMMDDHHMM (in GMT)
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    # Build the search query using the title and the submittedDate filter.
    # Example: ti:medical+AND+submittedDate:[start+TO+end]
    query = f"ti:{title}+AND+submittedDate:[{start_str}+TO+{end_str}]"
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"

    print("Query URL:", url)
    feed = feedparser.parse(url)
    return feed.entries

def download_pdf(pdf_url: str, folder: str, filename: str):
    """
    Download a PDF from the provided URL and save it to the specified folder with the given filename.

    Parameters:
        pdf_url (str): URL of the PDF file.
        folder (str): Local directory to save the PDF.
        filename (str): The filename for the downloaded PDF.
    """
    os.makedirs(folder, exist_ok=True)
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            filepath = os.path.join(folder, filename)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {pdf_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")

def download_medical_articles():
    """
    Download medical articles from arXiv as PDFs using a title filter and a submitted date range.
    PDFs are saved in the "papers" folder.

    Returns:
        A list of filenames for the downloaded PDFs.
    """
    # Define a date range (example: January 2024)
    start_date = datetime(2024, 1, 1, 0, 0)
    end_date   = datetime(2024, 1, 31, 23, 59)

    # TODO: for real run use at least 200
    articles = fetch_arxiv_articles("medical", start_date, end_date, max_results=10)

    pdf_folder = "papers"
    downloaded_files = []

    # clear the papers folder before downloading new PDFs
    clear_folder(pdf_folder)
    print(f"cleaned {pdf_folder} folder")

    for entry in articles:
        # Look for the PDF link among the entry's links.
        pdf_url = None
        for link in entry.links:
            if 'title' in link and link.title.lower() == 'pdf':
                pdf_url = link.href
                break

        if pdf_url:
            arxiv_id = entry.id.split('/abs/')[-1]
            pdf_filename = arxiv_id.replace('/', '_') + ".pdf"
            download_pdf(pdf_url, pdf_folder, pdf_filename)
            downloaded_files.append(pdf_filename)
        else:
            print("No PDF link found for entry:", entry.title)

    print(f"Downloaded {len(downloaded_files)} PDFs.")
    return downloaded_files
