class Hashable:
    """
    A class that represents an object that can be hashed and compared for equality.

    Attributes:
        counter (int): A counter used to assign unique hash values to each instance.

    Methods:
        __init__: Initializes a new instance of the Hashable class.
        __hash__: Returns the hash value of the instance.
        __eq__: Compares the instance with another object for equality.

    """

    counter = 0

    def __init__(self):
        self.hash = Hashable.counter
        Hashable.counter += 1

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash
