#!/usr/bin/env python

import sys
import cv2
import urllib
import os.path

import db
import extractors

# Name of temp file for saving download images.
WORKSPACE_PATH = "workspace/"

def run_extractor(conn, cur, name, function, img, imgID):
    """
    Runs the extractor with the given feature name and the given 
    extraction function on the given image. Saves the result to
    the database.
    """
    names = name() 
    print "Running extractors for {}".format(names)
    features = function(img)
    if isinstance(names, basestring):
        cur.execute("UPDATE features SET {} = {} WHERE id = {}".format(names, features, imgID))
    elif isinstance(names, list):
        for name, feature in zip(names, features):
            cur.execute("UPDATE features SET {} = {} WHERE id = {}".format(name, feature, imgID))
    conn.commit()

def run_extractors(conn, cur, img, imgID):
    """
    Runs all feature extractors on the given image, and updates the
    image's row in the database.
    """
    for name, function in zip(extractors.names, extractors.functions):
        run_extractor(conn, cur, name, function, img, imgID)

def get_image_data(url):
    """
    Fetches an image's actual (RGB) data, given its URL.
    """
    img_name_index = url.rfind("/")
    img_name = url[img_name_index + 1:]
    print "Processing {}...".format(img_name)
    if not os.path.isfile(WORKSPACE_PATH + img_name):
        urllib.urlretrieve(url, WORKSPACE_PATH + img_name)
    return cv2.imread(WORKSPACE_PATH + img_name)

def update_images(conn, cur):
    """
    Walks through each image in the database and recalculates all of
    its features.
    """
    cur.execute("SELECT id, url FROM features;")
    rows = cur.fetchall()
    for row in rows:
        imgID = row[0]
        url = row[1]
        img = get_image_data(url);
        run_extractors(conn, cur, img, imgID)

def main(args):
    """
    Right now what this does is loop through each image in the image
    database, run every feature extractor, and then update the
    corresponding image's feature information.
    """
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./extractor.py [optional: feature name]\n
              If a feature name is provided, only that feature will 
              be updated.
              """
        return

    conn, cur = db.connect()
    update_images(conn, cur)

if __name__ == "__main__":
    main(sys.argv)