#!/usr/bin/env python

import sys
import sqlite3
import extractors

# Name of sqlite database file holding our image features
DB_FILENAME = "features.db"

def run_extractor(cur, name, function, img):
	"""
	Runs the extractor with the given name and extraction function on
	the given image. If a column corresponding to the given name does
	not yet create in the database, it is created now. The result is
	then stored in the image's row, at that column.
	"""
	# TODO: Some kind of ALTER TABLE call to add the column
	# SQLite doesn't have an alter table??
	function(img)
	pass

def run_extractors(cur, img):
	"""
	Runs all feature extractors on the given image, and updates the
	image's row in the database.
	"""
	for name, function in zip(extractors.names, extractors.functions):
		run_extractor(name, function, img)

def get_image_data(url):
	"""
	Fetches an image's actual (RGB) data, given its URL.
	"""
	pass

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
		run_extractors(img)

def connect_to_db(filename):
	"""
	Connects to a sqlite3 database stored at the given filename.
	Returns a connection object and a cursor object.
	"""
	conn = sqlite3.connect(filename)
	cur = conn.cursor()
	return conn, cur

def main(args):
	"""
	Right now what this does is loop through each image in the image
	database, run every feature extractor, and then update the
	corresponding image's feature information.
	"""
	if len(args) == 2 and args[1] == "help":
		print "Usage: ./fe_manager.py"
		return

	conn, cur = connect_to_db(DB_FILENAME)
	cur.execute("CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT);");

	update_images(cur)

if __name__ == "__main__":
	main(sys.argv)