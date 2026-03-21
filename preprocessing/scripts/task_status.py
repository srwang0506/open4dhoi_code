"""
Stub task_status module for standalone usage.
The original module reports progress to a central server; this stub is a no-op.
"""


class TaskProgress:
    """No-op progress reporter."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    def fail(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
