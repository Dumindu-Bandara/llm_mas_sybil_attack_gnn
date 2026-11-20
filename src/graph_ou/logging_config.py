import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s %(levelname)-8s %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "",
            "maxBytes": 10_000_000,     # 10 MB per file
            "backupCount": 5,           # keep app.log.1 ... app.log.5
            "encoding": "utf-8",
        },
    },
    # Handlers live ONLY here
    "root": {
        "level": "WARNING",
        "handlers": ["console", "file"],
    },
    # Levels only: no "handlers" key, propagate stays True (the default)
    "loggers": {
        "graph_ou": {"level": "DEBUG"},
        "__main__": {"level": "DEBUG"},
    },
}

def setup_logging(log_file: str) -> None:
    LOGGING["handlers"]["file"]["filename"] = log_file
    logging.config.dictConfig(LOGGING)