#!/usr/bin/env python

import sys
import locale
import db

from urllib2 import urlopen, URLError
from bs4 import BeautifulSoup

class DAImage:
    """
    Represents a deviantART image and its associated metadata.
    """
    base_url = ""
    url = ""
    medium = ""
    favs = 0
    views = 0
    comments = 0
    # watchers = 0

def write_img_to_db(conn, cur, img):
    """
    Given a DAImage object and a database connection/cursor, 
    write the image information to a database row. If a row
    already exists for that image (by comparing URLs), then
    update the existing row instead.
    """
    query = "SELECT id FROM features WHERE url = '{}'".format(img.url)
    cur.execute(query)
    rows = cur.fetchall()
    if len(rows) == 0:
        query = """
                INSERT INTO features (base_url, url, views, favorites, medium)  
                VALUES ('{}', '{}', '{}', '{}', '{}')
                """.format(img.base_url, img.url, img.views, img.favs, img.medium);
    else:
        query = """
                UPDATE features
                SET views = '{}',
                    favorites = '{}',
                    medium = '{}'
                WHERE id = '{}'
                """.format(img.views, img.favs, img.medium, rows[0]);
    cur.execute(query)
    conn.commit()

def update_img_metadata(conn, cur):
    """
    Updates the view/favs/medium metadata for every image in
    the database with the given connection/cursor.
    """
    query = "SELECT base_url FROM features"
    cur.execute(query)
    rows = cur.fetchall()
    for (base_url,) in rows:
        img_page_html = get_page_html(base_url)
        if (img_page_html != None):
            img = scrape_img_page(img_page_html, base_url)
            write_img_to_db(conn, cur, img)

def get_page_html(page_url):
    """
    Returns an HTML string for the given page URL.
    """
    try:
        return urlopen(page_url)
    except URLError, e:
        print "URLError: " + str(e)
        return None

def scrape_artist_stats_page(page_html):
    """
    Scrapes an HTML page corresponding to an artist stats
    from deviantART.
    """
    sp = BeautifulSoup(page_html, "html5lib")

    stats = sp.find(id="ViewsTotalDiv")
    watchers = stats.children[0]
    return watchers

def scrape_img_page(page_html, page_url):
    """
    Scrapes an HTML page corresponding to a single
    image on deviantART. Returns a DAImage object with
    the image information.
    """
    sp = BeautifulSoup(page_html, "html.parser")

    img = DAImage()
    img.base_url = page_url
    img.url = sp.find("img", "dev-content-normal").get("src")

    stats = sp.find("div", "dev-right-bar-content dev-metainfo-content dev-metainfo-stats")
    img.views = locale.atoi(stats.dl.contents[1].contents[0])
    img.favs = locale.atoi(stats.dl.contents[3].contents[0])
    img.comments = locale.atoi(stats.dl.contents[5].contents[0])

    nav_breadcrumbs = sp.find("span", "dev-about-breadcrumb")
    img.medium = nav_breadcrumbs.span.a.span.contents[0]

    # Watchers information is generated via Javascript, can't scrape it. :(
    # about = sp.find("div", "dev-title-container")
    # artist_link = about.span.a.get("href")
    # artist_stats_link = artist_link + "stats/gallery/"
    # artist_stats_page_html = get_page_html(artist_stats_link)
    # img.watchers = scrape_artist_stats_page(artist_stats_page_html)

    print (img.url, img.views, img.favs, img.comments, img.medium)
    return img

def scrape_results_page(page_html):
    """
    Scrapes an HTML page corresponding to a deviantART
    search.
    """
    sp = BeautifulSoup(page_html, "html.parser")
    img_pages = sp.find_all("a", "thumb")
    imgs = []
    for link in img_pages:
        page_url = link.get("href")
        img_page_html = get_page_html(page_url)
        if (img_page_html != None):
            imgs.append(scrape_img_page(img_page_html, page_url))
    return imgs

def main(args):
    if len(args) == 2 and args[1] == 'help':
        print """
              Usage: ./scraper.py [-r | -s | -u] [url]

              Use -r if you want to pass a URL to a deviantART search
              results page. 

              Use -s if you want to pass a URL to a single
              deviantART image.

              Use -u if you want to update view/favs/medium information
              for existing images in the database.
              """
        return
    elif (args[1] == '-r' or args[1] == '-s') and len(args) < 3 :
        print "Insufficient arguments. Try 'help' for more information";
        return

    # For parsing number strings that have commas
    # locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    locale.setlocale( locale.LC_ALL, 'English_United States.1252' )

    # Handle to the image information database
    conn, cur = db.connect()

    # Scrape away...woohoo...
    if (args[1] == '-r'):
        page_html = get_page_html(args[2])
        imgs = scrape_results_page(page_html)
        for img in imgs:
            write_img_to_db(conn, cur, img)
    elif (args[1] == "-s"):
        page_html = get_page_html(args[2])
        img = scrape_img_page(page_html, args[2])
        write_img_to_db(conn, cur, img)
    elif (args[1] == "-u"):
        update_img_metadata(conn, cur)
    else:
        print "Unrecognized argument {}".format(args[1])

if __name__ == "__main__":
    main(sys.argv)