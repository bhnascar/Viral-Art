#!/usr/bin/env python

import sys
import locale

from urllib2 import urlopen, URLError
from bs4 import BeautifulSoup

class DAImage:
    """
    Represents a deviantART image and its associated metadata.
    """
    url = ""
    category = ""
    medium = ""
    favs = 0
    views = 0
    comments = 0
    watchers = 0

def get_page_html(page_url):
    """
    Returns an HTML string for the given page URL.
    """
    try:
        return urlopen(page_url)
    except URLError, e:
        print "URLError: " + str(e)
        return None

def scrape_img_page(page_html):
    """
    Scrapes an HTML page corresponding to a single
    image on deviantART.
    """
    sp = BeautifulSoup(page_html, "html.parser")

    img = DAImage()
    img.url = sp.find("img", "dev-content-normal").get("src")

    stats = sp.find("div", "dev-right-bar-content dev-metainfo-content dev-metainfo-stats")
    img.views = locale.atoi(stats.dl.contents[1].contents[0])
    img.favs = locale.atoi(stats.dl.contents[3].contents[0])
    img.comments = locale.atoi(stats.dl.contents[5].contents[0])
    print (img.url, img.views, img.favs, img.comments)

def scrape_results_page(page_html):
    """
    Scrapes an HTML page corresponding to a deviantART
    search.
    """
    sp = BeautifulSoup(page_html, "html.parser")
    img_pages = sp.find_all("a", "thumb")
    for link in img_pages:
        page_url = link.get("href")
        img_page_html = get_page_html(page_url)
        if (img_page_html != None):
            scrape_img_page(img_page_html)

def main(args):
    if len(args) == 2 and args[1] == 'help':
        print """
              Usage: ./scraper.py [-r | -s] [url]

              Use -r if you want to pass a URL to a deviantART search
              results page. 

              Use -s if you want to pass a URL to a single
              deviantART image.
              """
        return
    elif len(args) < 3:
        print "Insufficient arguments. Try 'help' for more information";
        return
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
    page_html = get_page_html(args[2])
    if (args[1] == '-r'):
        scrape_results_page(page_html)
    elif (args[1] == "-s"):
        scrape_img_page(page_html)
    else:
        print "Unrecognized argument {}".format(args[1])

if __name__ == "__main__":
    main(sys.argv)