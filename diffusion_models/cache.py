import weakref

class WeakKey(object):
  """Weak pointer equality based keys for hashable dicts. Does not keep x alive."""
  def __init__(self, x):
    self.id = id(x)
    self.weak = weakref.ref(x)
  def __hash__(self):
    return hash(self.id)
  def __eq__(self, other):
    a = self.weak()
    b = other.weak()
    return self.id == other.id and (a is b)

class WeakCache(object):
    """A cache using weak references so values are cached only as long as they are referenced from elsewhere."""
    def __init__(self, f):
        self.cache = {}
        self.f = f

    def clear(self):
        self.cache.clear()

    def __call__(self, x):
        """Look up the cached value of f(x)."""
        key = WeakKey(x)
        if key in self.cache:
          val = self.cache[key]()
          if val is not None:
            return val
        val = self.f(x)
        self.cache[key] = weakref.ref(val)
        return val
