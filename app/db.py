import os

import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USERNAME") or os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "beshow"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )