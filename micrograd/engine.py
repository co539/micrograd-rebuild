class Value:
    def __init__(self, data, _parents=(), _op =''):
        self.data = data
        self._parents = set(_parents)
        self._op = _op

        self.grad = 0.0
        self._backpropagate = lambda : None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data,(self, other),"+")

        def _backpropagate():
            self.grad += output.grad * 1.0 #accumulative due to chain rule
            other.grad += output.grad * 1.0
        output._backpropagate = _backpropagate

        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data,(self, other),"×")

        def _backpropagate():
            self.grad += output.grad * other.data  #accumulative due to chain rule
            other.grad += output.grad * self.data
        output._backpropagate = _backpropagate

        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        output = Value(self. data**other, (self, ), f'**{other}')

        def backpropagate():
            self.grad += output.grad * (other * self.data ** (other - 1))
        output._backpropagate = backpropagate

        return output

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backpropagate():
            self.grad += (out.data > 0) * out.grad

        out._backpropagate= _backpropagate

    def backpropagate(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        for node in topo:
            node.grad = 0.0 #resets all node grads to zero

        self.grad = 1.0

        for node in reversed(topo):
            node._backpropagate()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data = {self.data}, grad={self.grad})"

    #1:40:00