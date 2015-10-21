_ON = True
_OFF = False
_AUTO = None
_values = {
    'on': _ON,
    'ON': _ON,
    _ON: _ON,
    'off': _OFF,
    'OFF': _OFF,
    _OFF: _OFF,
    'auto': _AUTO,
    'AUTO': _AUTO,
    _AUTO: _AUTO,
}
_reprs = {
    _ON: 'ON',
    _OFF: 'OFF',
    _AUTO: 'AUTO',
}

_caches = {}


class Flag(object):

    """Ternary flag object for variables.

    The flag object represents either of ON, OFF, or AUTO. The flag object is
    _cached_, and can be compared by ``is`` operator.

    """
    def __new__(cls, name):
        if name in _flags:
            return _flags[name]
        flag = super(Flag, cls).__new__(cls)
        flag.value = _values[name]
        return flag

    def __bool__(self):
        value = self.value
        if value is _AUTO:
            raise TypeError('Flag AUTO cannot be converted to boolean')
        return value

    __nonzero__ = __bool__

    def __reduce__(self):
        return Flag, (self.value,)

    def __repr__(self):
        return _reprs[self.value]

    def __eq__(self, other):
        return self is Flag(other)

    def __hash__(self):
        return hash(self.value)


_flags = {}
ON = Flag('ON')
OFF = Flag('OFF')
AUTO = Flag('AUTO')

_flags = {
    'on': ON,
    'ON': ON,
    _ON: ON,
    'off': OFF,
    'OFF': OFF,
    _OFF: OFF,
    'auto': AUTO,
    'AUTO': AUTO,
    _AUTO: AUTO,
}


def aggregate_flags(flags):
    """Returns an aggregated flag given a sequence of flags.

    If both ON and OFF are found, this function raises an error. Otherwise,
    either of ON and OFF that appeared is returned. If all flags are AUTO, then
    it returns AUTO.

    Args:
        flags (sequence of Flag): Input flags.

    Returns:
        Flag: The result of aggregation.

    """
    on = any([flag is ON for flag in flags])
    off = any([flag is OFF for flag in flags])
    if on:
        if off:
            raise ValueError('ON and OFF flags cannot be mixed.')
        else:
            return ON
    else:
        return OFF if off else AUTO