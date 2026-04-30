import pymongo
import logging

################
#   MONGO DB   #
################

from pymongo import MongoClient

def connect_db(
        connection_uri: str,
):
    """
    Parameters
    ----------
    connection_uri : str
        e.g.
            "mongodb://localhost:27017/"
            "mongodb+srv://<username>:<password>@cluster.mongodb.net/"

    Returns
    -------
    client : MongoClient

    """
    # For a local database
    client = MongoClient(connection_uri)

    # Verify connection with a ping
    try:
        client.admin.command('ping')
        logging.info("Connected to database successfully!")
    except Exception as e:
        logging.error(f"Database Connection failed: {e}")

    return client