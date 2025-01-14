import requests
from urllib.request import quote, unquote
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm


def search_query_in_internet(query=None, url=None):
    """
    Performs a web search for a query and retrieves the results.

    Args:
        query (str): The search query.
        url (str): An optional URL to start the search from. Defaults to None.

    Returns:
        Response: The response object containing the search results.
    """
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/130.0.2849.80"}
    params = {'hl': 'ru', 'gl': 'ru'}

    if query is not None:
        query += " site:.ru"
        encoded_query = quote(query)

    if url is None:
        url = f"https://www.google.com/search?q={encoded_query}"

    response = requests.get(url=url, headers=headers, verify=False)
    response.encoding = response.apparent_encoding

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        exit()

    return response


def extract_links_from_response(response, excluded_domains, top_k=5):
    """
    Extracts external links from a search result page response.

    Args:
        response (Response): The response object containing the search results.
        excluded_domains (list): A list of domains to exclude from the results.
        top_k (int): The number of top links to return.

    Returns:
        list: A list of cleaned external links.
    """
    soup = BeautifulSoup(response.content, "html.parser")
    all_links = [a.get("href") for a in soup.find_all("a", href=True)]

    external_links = []
    for link in all_links:
        if link.startswith("/url?q=") or link.startswith("http"):
            url_part = link.split('&')[0]
            actual_url = url_part.split("/url?q=")[-1]
            actual_url = unquote(actual_url)

            if actual_url.startswith("http"):
                external_links.append(actual_url)

    cleaned_links = clean_links(external_links, excluded_domains)

    return cleaned_links[:top_k]


def get_text(url):
    """
    Retrieves the text content from a webpage.

    Args:
        url (str): The URL of the webpage.

    Returns:
        str: The cleaned text content from the webpage.
    """
    response = search_query_in_internet(url=url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_text = soup.text
    return re.sub(r'[\n]+', '\n', raw_text.replace('\xa0', ' '))


def internet_search(query, k=2):
    """
    Performs a web search for a query and fetches the relevant texts.

    Args:
        query (str): The query string to search for.
        k (int): The number of results to fetch. Defaults to 2.

    Returns:
        list: A list of texts extracted from the search results.
    """
    excluded_domains = ['google', 'gstatic']
    response = search_query_in_internet(query=query)
    links = extract_links_from_response(response, excluded_domains, top_k=k)

    all_texts = []
    for link in tqdm(links, desc="Fetching links"):
        time.sleep(2)
        all_texts.append(get_text(url=link))

    return all_texts
