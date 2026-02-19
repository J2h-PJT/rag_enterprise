from sqlalchemy import create_engine
from config import PG_CONN_STR

def create_db_engine():
    return create_engine(
        PG_CONN_STR,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        connect_args={"options": "-c client_encoding=UTF8"},
    )
