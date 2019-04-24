# Improved numpy's `einsum`

## Example

```python
import numpy
from einsum import einsum

X = numpy.random.random((3, 4, 5))
Y = numpy.random.random((4, 6))

# numpy's einsum
# (you can use only one character for each index name.)
Z1 = numpy.einsum('ijk, jl -> ikl', X, Y)

# improved einsum
# (you can use more than one character for index names.)
Z2 = einsum('i1,i2,i3; i2,kkkk -> i1,i2,kkkk', X, Y)
```

## License

MIT
