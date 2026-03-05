import logging

import jira

__all__ = ('ToUsername', 'determine_username',)


class ToUsername:
    _jira_instance: jira.JIRA
    _id_to_username_mapping: dict[str, str]

    def __init__(self, jira_instance: jira.JIRA):
        self._jira_instance = jira_instance
        self._id_to_username_mapping = {}

    def __call__(self, value):
        """Converts the given value to a username (user ID), if needed, while caching the result.

        Args:
            value (str/jira.User): String (account ID or user ID) or User instance

        Returns:
            str: The corresponding username
        """
        user_id = value.accountId if hasattr(value, 'accountId') else str(value)
        if user_id in self._id_to_username_mapping:
            return self._id_to_username_mapping[user_id]

        if not isinstance(value, str):
            self._id_to_username_mapping[user_id] = determine_username(value)
        elif len(value) >= 24:  # accountId
            user = self._jira_instance.user(user_id)
            self._id_to_username_mapping[user_id] = determine_username(user)
        return self._id_to_username_mapping.get(user_id, value)


def determine_username(user: jira.User):
    """Determines the username (user ID) for the given User.

    Args:
        user (jira.User): User instance

    Returns
        str: Corresponding username

    Raises:
        Exception: Failed to determine username
    """
    if getattr(user, 'emailAddress', ''):
        username = user.emailAddress.split('@')[0]
    elif getattr(user, 'name', ''):  # compatibility with Jira Server
        username = user.name
    elif getattr(user, 'displayName', ''):
        full_name = user.displayName
        username = f'"{full_name}"'
        logging.error(f"Failed to fetch email address of {full_name!r}: they restricted its visibility; "
                      f"using identifier {username!r} as fallback value.")
    else:
        raise Exception(f"Failed to determine username of {user}")
    return username
