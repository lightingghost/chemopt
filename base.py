import tensorflow as tf

class Error(Exception):
    pass
    
class NotConnectedError(Error):
    pass

class ParentNotBuiltError(Error):
    pass

class IncompatibleShapeError(Error):
    pass

class UnderspecifiedError(Error):
    pass

class NotSupportedError(Error):
    pass
