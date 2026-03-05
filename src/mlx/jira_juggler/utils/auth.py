import logging
from getpass import getpass

from decouple import config


__all__ = ('fetch_credentials',)


def fetch_credentials():
    """ Fetches the credentials from the .env file by default or, alternatively, from the user's input

    Returns:
        str: email address or username
        str: API token or password
    """
    username = config('JIRA_USERNAME', default='')
    api_token = config('JIRA_API_TOKEN', default='')
    if not username:
        username = input('JIRA email address (or username): ')
    if not api_token:
        password = config('JIRA_PASSWORD', default='')
        if password:
            logging.warning('Basic authentication with a JIRA password may be deprecated. '
                            'Consider defining an API token as environment variable JIRA_API_TOKEN instead.')
            return username, password
        else:
            api_token = getpass(f'JIRA API token (or password) for {username}: ')
    return username, api_token
