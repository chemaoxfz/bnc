
# How to install `BCN` package
⚠️ Currently, this python package only can be installed from GitHub by running command:

`pip install git+https://github.com/chemaoxfz/bnc.git`

# First starting
## Build up a binding netwrok
The simplest binding network consists of a single binding reaction
$$
E + S \leftrightharpoons C
$$
representing the complexing of two species.

While it is possible to obtain a simple network (from the point of view of the number of unique species) by asserting, for example, that $E = S$, we will keep the number of species to $3$ to preserve generality.
$$
\begin{aligned}
x &= (E, S, C) \\\\
q &= (q_E, q_S) \\\\
N &= \begin{bmatrix}1\ 1\ -1\end{bmatrix} \\\\
L &= \begin{bmatrix}1\ 0\ 1 \\\\ 0\ 1\ 1\end{bmatrix}
\end{aligned}
$$
Then, we obtain a binding network,

```python title='bn.py'
import numpy as np
from bcn.binding_network import binding_network


l_mat = np.array([
  [1, 0, 1],
  [0, 1, 1]
])

n_mat = np.array([
  [1, 1, -1]
])

bn = binding_network(n_mat=n_mat, l_mat=l_mat)
```

## Build up a catalysis network

```python title='cn.py'
import numpy as np
from bcn.catalysis_network import catalysis_network


s_mat = np.array([
    [-1], # S
    [1],  # P
])

cn = catalysis_network(s_mat)
```