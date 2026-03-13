import os
from functools import lru_cache
from pathlib import Path


DEFAULT_USER_DATA_DIRNAME = ".app_data"
DEFAULT_UPLOAD_DIRNAME = "uploads"
DEFAULT_LOG_DIRNAME = "logs"
DEFAULT_VECTOR_DB_DIRNAME = ".chroma"


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    configured_root = os.getenv("APP_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_app_root() -> Path:
    return get_project_root()


@lru_cache(maxsize=1)
def get_user_data_root() -> Path:
    configured_root = os.getenv("APP_USER_DATA_DIR")
    if configured_root:
        return ensure_directory(Path(configured_root).expanduser().resolve())
    return ensure_directory(get_app_root() / DEFAULT_USER_DATA_DIRNAME)


def get_upload_dir() -> Path:
    configured_dir = os.getenv("APP_UPLOAD_DIR")
    if configured_dir:
        return ensure_directory(Path(configured_dir).expanduser().resolve())
    return ensure_directory(get_user_data_root() / DEFAULT_UPLOAD_DIRNAME)


def get_log_dir() -> Path:
    configured_dir = os.getenv("APP_LOG_DIR")
    if configured_dir:
        return ensure_directory(Path(configured_dir).expanduser().resolve())
    return ensure_directory(get_user_data_root() / DEFAULT_LOG_DIRNAME)


def get_vector_db_dir(path_value: str | Path | None = None) -> Path:
    configured_path = path_value if path_value is not None else os.getenv(
        "VECTOR_DB_PATH",
        DEFAULT_VECTOR_DB_DIRNAME,
    )
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return ensure_directory(path)
    return ensure_directory(get_user_data_root() / path)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return get_project_root() / path


def to_relative_display_path(path_value: str | Path, *, base_path: Path | None = None) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()

    if base_path is not None:
        try:
            return path.relative_to(base_path).as_posix()
        except ValueError:
            pass

    try:
        return path.relative_to(get_project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_path_caches() -> None:
    get_project_root.cache_clear()
    get_user_data_root.cache_clear()
