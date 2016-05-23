#!/usr/bin/env python

import sys
import cv2
import urllib
import os.path

import db
import extractors

# Name of temp file for saving download images.
WORKSPACE_PATH = "workspace/"

def run_extractor(conn, cur, name_fn, feature_fn, img, imgID, filters):
    """
    Runs the extractor with the given feature name and the given 
    extraction function on the given image. Saves the result to
    the database.
    """
    names = name_fn()

    # Handle single features
    if isinstance(names, basestring):
        if filters and names not in filters:
            return
        
        print "Running extractor for {}".format(names)
        features = feature_fn(img)
        if features == None:
            features = "null"
        cur.execute("UPDATE features SET {} = {} WHERE id = {}".format(names, features, imgID))
    
    # Handle group of related features
    elif isinstance(names, list):
        if filters:
            found = sum(1 for name in names if name in filters)
            if found == 0:
                return

        print "Running extractors for {}".format(names)
        features = feature_fn(img)
        for (name, feature) in zip(names, features):
            if feature == None:
                feature = "null"
            cur.execute("UPDATE features SET {} = {} WHERE id = {}".format(name, feature, imgID))
    
    conn.commit()

def run_extractors(conn, cur, img, imgID, filters):
    """
    Runs all feature extractors on the given image, and updates the
    image's row in the database.
    """
    for name_fn, feature_fn in zip(extractors.names, extractors.functions):
        run_extractor(conn, cur, name_fn, feature_fn, img, imgID, filters)

def get_image_data(url):
    """
    Fetches an image's actual (RGB) data, given its URL.
    """
    img_name_index = url.rfind("/")
    img_name = url[img_name_index + 1:]
    if not os.path.isfile(WORKSPACE_PATH + img_name):
        urllib.urlretrieve(url, WORKSPACE_PATH + img_name)
    return cv2.imread(WORKSPACE_PATH + img_name)

def update_images(conn, cur, filters = None):
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
        print "Processing {}...".format(url)
        run_extractors(conn, cur, img, imgID, filters)

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

    if len(args) == 2:
        update_images(conn, cur, args[1])
    else:
        update_images(conn, cur)

if __name__ == "__main__":
    main(sys.argv)