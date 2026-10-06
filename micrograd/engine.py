class Value:
    def __init__(self, data, _parents=(), _op =''):
        self.data = data
        self._parents = set(_parents)
        self._op = _op

        self.grad = 0.0  # gradient is accumulative
        self._backpropagate = lambda : None

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data,(self, other),"+")

        def _backpropagate():
            self.grad = 1.0 * output.grad
            other.grad = 1.0 * output.grad

        output._backpropagate = _backpropagate()

        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data,(self, other),"×")

        def _backpropagate():
            self.grad = other.data * output.grad
            other.grad = self.data * output.grad

        output._backpropagate = _backpropagate()

        return output

    #1:20:00