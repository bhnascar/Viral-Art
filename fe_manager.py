#!/usr/bin/env python

import sys
import cv2
import sqlite3
import extractors
import urllib

# Name of sqlite database file holding our image features
DB_FILENAME = "features.db"

# Name of temp file for saving download images.
TMP_IMG_FILENAME = "workspace/temp.tmp"

# Default output file for features.
DEFAULT_OUTPUT_FILE = "workspace/features.txt"

def run_extractor(cur, name, function, img):
    """
    Runs the extractor with the given name and extraction function on
    the given image. If a column corresponding to the given name does
    not yet create in the database, it is created now. The result is
    then stored in the image's row, at that column.
    """
    # TODO: insert result into the correct database column
    print function(img)

def run_extractors(cur, img):
    """
    Runs all feature extractors on the given image, and updates the
    image's row in the database.
    """
    for name, function in zip(extractors.names, extractors.functions):
        run_extractor(cur, name, function, img)

def get_image_data(url):
    """
    Fetches an image's actual (RGB) data, given its URL.
    """
    urllib.urlretrieve(url, TMP_IMG_FILENAME)
    return cv2.imread(TMP_IMG_FILENAME)

def update_images(cur):
    """
    Walks through each image in the database and recalculates all of
    its features.
    """
    cur.execute("SELECT url FROM features;")
    rows = cur.fetchall()
    for row in rows:
        url = row[0]
        img = get_image_data(url);
        run_extractors(cur, img)

def dump_features(cur, outputfile = DEFAULT_OUTPUT_FILE):
    """
    Reads features for every image out of the features database and
    writes it to a CSV file where each row corresponds to a single
    feature vector.
    """
    cur.execute("SELECT * FROM features")
    with open(outputfile, "a") as output:
        for row in rows:
            output.write(",".join(map(str, row[2:])))

def connect_to_db(filename):
    """
    Connects to a sqlite3 database stored at the given filename.
    Returns a connection object and a cursor object.
    """
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    return conn, cur

def validate_db_structure(cur):
    """
    Validates that all the 'features' table exists in the database.
    Validates that the required feature columns exist in the
    features table. Creates any columnsthat are missing.
    """
    cur.execute("CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT);");
    cur.execute("PRAGMA table_info(features)")
    rows = cur.fetchall()
    column_names = [row[1] for row in rows]
    for feature_name in extractors.names:
        if feature_name() not in column_names:
            cur.execute("AlTER TABLE features ADD COLUMN ? TEXT;", feature_name())

def main(args):
    """
    Right now what this does is loop through each image in the image
    database, run every feature extractor, and then update the
    corresponding image's feature information.
    """
    if len(args) == 2 and args[1] == "help":
        print "Usage: ./fe_manager.py"
        return

    # Connect to DB and validate its structure.
    conn, cur = connect_to_db(DB_FILENAME)
    # validate_db_structure(cur)

    # Update images.
    update_images(cur)

if __name__ == "__main__":
    main(sys.argv)