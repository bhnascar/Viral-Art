#!/usr/bin/env python

import sys
import os.path

import db
import extractors

# Default output file for features.
DEFAULT_OUTPUT_FILE = "output/features.txt"

def dump_features(cur, features = "", outputfile = DEFAULT_OUTPUT_FILE):
    """
    Reads features for every image out of the features database and
    writes it to a CSV file where each row corresponds to a single
    feature vector.
    """
    # Write column names
<<<<<<< HEAD
    # if not os.path.isfile(outputfile):
    with open(outputfile, "w") as output:
=======
    with open(outputfile, "w+") as output:
>>>>>>> 511b963f29812cc78a151340875a3bbd33bc4b0e
        cur.execute("PRAGMA table_info(features);")
        rows = cur.fetchall()
        if not features:
            column_names = [row[1] for row in rows]
        else:
<<<<<<< HEAD
            column_names = [row[1] for row in rows if row[1] in features]
=======
            column_names = [row[1] for row in rows if row[1] not in features]
>>>>>>> 511b963f29812cc78a151340875a3bbd33bc4b0e
        output.write(",".join(column_names) + "\n")

    # Write row data
    selectedColumns = ",".join(column_names)
    cur.execute("SELECT {} FROM features;".format(selectedColumns))
    rows = cur.fetchall()
    with open(outputfile, "a") as output:
        for row in rows:
            output.write(",".join(map(str, row)) + "\n")

def main(args):
    """
    Right now what this does is loop through each image in the image
    database, run every feature extractor, and then update the
    corresponding image's feature information.
    """
    if len(args) == 2 and args[1] == "help":
        print """
              Usage: ./dumper.py [optional: filename features]\n
              By default, this dumps all features to the default
              output file under the workspace directory. If you
              like you can specify the output file and a comma-
              separated list of specific features to exclude. \n
              Ex: ./dumper.py output.txt Average_Saturation,Average_Hue
              """
        return

    conn, cur = db.connect()

    if len(args) > 2:
        dump_features(cur, args[2], args[1])
    else:
        dump_features(cur)

if __name__ == "__main__":
    main(sys.argv)