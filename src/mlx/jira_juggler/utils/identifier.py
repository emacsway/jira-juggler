__all__ = ('to_identifier',)


def to_identifier(key):
    """Converts given key to identifier, interpretable by TaskJuggler as a task-identifier

    Args:
        key (str): Key to be converted

    Returns:
        str: Valid task-identifier based on given key
    """
    return key.replace('-', '_')
