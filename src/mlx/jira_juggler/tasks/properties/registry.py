from mlx.jira_juggler.utils.identifier import to_identifier

__all__ = ('Registry',)


class Registry(dict):
    def path(self, key):
        if key not in self:
            return key
        path = []
        task = self[key]
        while task:
            path.append(to_identifier(task.key))
            task = task.parent
        path.reverse()
        return ".".join(path)
