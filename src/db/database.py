from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.db.models import Base


@lru_cache(maxsize=1)
def _get_engine():
    # Lazy + cached so config.DB_PATH can be overridden (e.g. in tests)
    # before the engine is first created.
    return create_engine(f"sqlite:///{config.DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    Base.metadata.create_all(_get_engine())


def get_session() -> Session:
    session_factory = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return session_factory()


def reset_engine_cache() -> None:
    _get_engine.cache_clear()
