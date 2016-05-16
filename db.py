import sqlite3

import extractors

# Name of sqlite database file holding our image features
DB_FILENAME = "features.db"

def validate_tables(cur):
    """
    Validates that the 'features' table exists in the database.
    """
    cur.execute("""
                CREATE TABLE IF NOT EXISTS 
                features (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          url TEXT,
                          views TEXT, 
                          favorites TEXT);
                """);

def validate_columns(cur):
    """
    Validates that the required feature columns exist in the
    features table. Creates any columnsthat are missing.
    """
    cur.execute("PRAGMA table_info(features)")
    rows = cur.fetchall()
    column_names = [row[1] for row in rows]
    for feature_name_fn in extractors.names:
        feature_name = feature_name_fn()

        # Single feature
        if isinstance(feature_name, basestring):
            if feature_name not in column_names:
                cur.execute("AlTER TABLE features ADD COLUMN {} TEXT".format(feature_name))
        
        # Several related features, ex. hue roughness, saturation roughness, etc.
        elif isinstance(feature_name, list):
            for sub_feature_name in feature_name:
                if sub_feature_name not in column_names:
                    cur.execute("AlTER TABLE features ADD COLUMN {} TEXT;".format(sub_feature_name))

def connect(filename = DB_FILENAME):
    """
    Connects to a sqlite3 database stored at the given filename.
    Returns a connection object and a cursor object.
    """
    conn = sqlite3.connect(filename)
    cur = conn.cursor()

    validate_tables(cur)
    validate_columns(cur)

    return conn, cur