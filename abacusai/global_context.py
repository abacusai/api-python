import threading


lock = threading.Lock()
_context = {}


def get(key):
    return _context.get(key)


def set(key, val):
    _context[key] = val
