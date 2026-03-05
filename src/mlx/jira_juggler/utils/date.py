__all__ = ('to_juggler_date',)


def to_juggler_date(date):
    """Converts given datetime.datetime object to a string that can be interpreted by TaskJuggler

    The resolution is 60 minutes.

    Args:
        date (datetime.datetime): Datetime object

    Returns:
        str: String representing the date and time in TaskJuggler's format
    """
    return date.strftime('%Y-%m-%d-%H:00-%z').rstrip('-')
