$$
P (\hat {y} | \mathbf {x}, z) = P (\hat {y} | \mathbf {x}), \tag {1}
$$

$$
P (\hat {y} = 1 | z = 0) = P (\hat {y} = 1 | z = 1), \tag {2}
$$

$$
P (\hat {y} \neq y | z = 0) = P (\hat {y} \neq y | z = 1), \tag {3}
$$

$$
P (\hat {y} \neq y | z = 0, y = - 1) = P (\hat {y} \neq y | z = 1, y = - 1), \quad (4)
$$

$$
P (\hat {y} \neq y | z = 0, y = 1) = P (\hat {y} \neq y | z = 1, y = 1), \tag {5}
$$

$$
P (\hat {y} \neq y | z = 0, \hat {y} = - 1) = P (\hat {y} \neq y | z = 1, \hat {y} = - 1), \tag {6}
$$

$$
P (\hat {y} \neq y | z = 0, \hat {y} = 1) = P (\hat {y} \neq y | z = 1, \hat {y} = 1). \tag {7}
$$

$$
P (\hat {y} \neq y | z = 0) - P (\hat {y} \neq y | z = 1) \geq - \epsilon ,
$$

$$
\begin{array}{l} \operatorname {C o v} (z, g _ {\theta} (y, \mathbf {x})) = \mathbb {E} [ (z - \bar {z}) (g _ {\theta} (y, \mathbf {x}) - \bar {g} _ {\theta} (y, \mathbf {x})) ] \\ \approx \frac {1}{N} \sum_ {(\mathbf {x}, y, z) \in \mathcal {D}} (z - \bar {z}) g _ {\boldsymbol {\theta}} (y, \mathbf {x}), \tag {9} \\ \end{array}
$$

$$
g _ {\boldsymbol {\theta}} (y, \mathbf {x}) = \min  (0, y d _ {\boldsymbol {\theta}} (\mathbf {x})), \tag {10}
$$

$$
g _ {\boldsymbol {\theta}} (y, \mathbf {x}) = \min  \left(0, \frac {1 - y}{2} y d _ {\boldsymbol {\theta}} (\mathbf {x})\right), \text {o r} \tag {11}
$$

$$
g _ {\boldsymbol {\theta}} (y, \mathbf {x}) = \min  \left(0, \frac {1 + y}{2} y d _ {\boldsymbol {\theta}} (\mathbf {x})\right), \tag {12}
$$

$$
\frac {1}{N} \sum_ {(\mathbf {x}, y, z) \in \mathcal {D}} (z - \bar {z}) g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \geq - c,
$$

$$
\sum_ {(\mathbf {x}, y, z) \in \mathcal {D}} (z - \bar {z}) g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \sim c, \tag {14}
$$

$$
\sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {0}} (0 - \bar {z}) g _ {\boldsymbol {\theta}} (y, \mathbf {x}) + \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} (1 - \bar {z}) g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \sim c, \tag {15}
$$

$$
\frac {- N _ {1}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {0}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) + \frac {N _ {0}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \sim c, \tag {16}
$$

$$
\begin{array}{l} + \frac {N _ {0}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \leq c \tag {17} \\ \frac {- N _ {1}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {0}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \\ + \frac {N _ {0}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \geq - c, \\ \end{array}
$$

$$
\begin{array}{l} + \frac {N _ {0}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \leq c \tag {18} \\ \frac {- N _ {1}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {0}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \\ + \frac {N _ {0}}{N} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {1}} g _ {\boldsymbol {\theta}} (y, \mathbf {x}) \geq - c. \\ \end{array}
$$

$$
D _ {F P R} = P (\hat {y} \neq y | z = 0, y = - 1) - P (\hat {y} \neq y | z = 1, y = - 1),
$$

$$
D _ {F N R} = P (\hat {y} \neq y | z = 0, y = 1) - P (\hat {y} \neq y | z = 1, y = 1),
$$

$$
p (\mathbf {x} | z = 0, y = 1) = \mathcal {N} ([ 2, 2 ], [ 3, 1; 1, 3 ])
$$

$$
p (\mathbf {x} | z = 1, y = 1) = \mathcal {N} ([ 2, 2 ], [ 3, 1; 1, 3 ])
$$

$$
p (\mathbf {x} \mid z = 0, y = - 1) = \mathcal {N} ([ 1, 1 ], [ 3, 3; 1, 3 ])
$$

$$
p (\mathbf {x} | z = 1, y = - 1) = \mathcal {N} ([ - 2, - 2 ], [ 3, 1; 1, 3 ]).
$$

$$
p (\mathbf {x} | z = 0, y = 1) = \mathcal {N} ([ 2, 0 ], [ 5, 1; 1, 5 ])
$$

$$
p (\mathbf {x} \mid z = 1, y = 1) = \mathcal {N} ([ 2, 3 ], [ 5, 1; 1, 5 ])
$$

$$
p (\mathbf {x} | z = 0, y = - 1) = \mathcal {N} ([ - 1, - 3 ], [ 5, 1; 1, 5 ])
$$

$$
p (\mathbf {x} | z = 1, y = - 1) = \mathcal {N} ([ - 1, 0 ], [ 5, 1; 1, 5 ])
$$

$$
p (\mathbf {x} | z = 0, y = 1) = \mathcal {N} ([ 1, 2 ], [ 5, 2; 2, 5 ])
$$

$$
p (\mathbf {x} \mid z = 1, y = 1) = \mathcal {N} ([ 2, 3 ], [ 1 0, 1; 1, 4 ])
$$

$$
p (\mathbf {x} \mid z = 0, y = - 1) = \mathcal {N} ([ 0, - 1 ], [ 7, 1; 1, 7 ])
$$

$$
p (\mathbf {x} | z = 1, y = - 1) = \mathcal {N} ([ - 5, 0 ], [ 5, 1; 1, 5 ])
$$

$$
\begin{array}{l} \text {I n c r e a s e p e n a l t y :} C = C + \Delta . \\ \boldsymbol {\theta} = \operatorname {a r g m i n} _ {\boldsymbol {\theta}} C \sum_ {\mathbf {d} \in \mathcal {P}} L (\boldsymbol {\theta}, \mathbf {d}) + \sum_ {\mathbf {d} \in \mathcal {P}} L (\boldsymbol {\theta}, \mathbf {d}) \\ \text {e n d} \end{array}
$$
