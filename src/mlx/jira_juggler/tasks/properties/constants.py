
__all__ = (
    'TODO_STATUSES',
    'PROGRESS_STATUSES',
    'DEVELOPED_STATUSES',
    'RESOLVED_STATUSES',
    'PENDING_STATUSES',
    'DONE_STATUSES',
    'TAB',
)

TODO_STATUSES = (
    'to do',
    'blocked',
    'reopened',
    'postponed',
)
PROGRESS_STATUSES = (
    'in progress',
)
DEVELOPED_STATUSES = (
    'ready for code review',
    'in code review',
)
RESOLVED_STATUSES = (
    'approved',
    'resolved',
    'merged to dev'
)
PENDING_STATUSES = (
    'in testing',
    'ready for testing on qa',
    'ready for deployment',
)
DONE_STATUSES = (
    'closed',
    'cancelled',
)
TAB = ' ' * 2
