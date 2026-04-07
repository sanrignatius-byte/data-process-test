$$
P (y = 1 | s = 1) = P (y = 1 | s = 0)
$$

$$
P (\eta (\mathbf {x}) = 1 | s = 1) = P (\eta (\mathbf {x}) = 1 | s = 0)
$$

$$
B E R (f (\mathcal {X}), \mathcal {S}) > \epsilon
$$

$$
B E R (f (\mathcal {X}), \mathcal {S}) = \frac {P [ f (\mathcal {X}) = 0 | \mathcal {S} = 1 ] + P [ f (\mathcal {X}) = 1 | \mathcal {S} = 0 ]}{2}.
$$

$$
\max  _ {D} \quad \mathbb {E} _ {\mathbf {x} \sim P _ {\mathrm {d a t a}}} [ \log D (\mathbf {x}) ] + \mathbb {E} _ {\mathbf {z} \sim P _ {\mathbf {z}}} [ \log (1 - D (G (\mathbf {z}))) ], \tag {1}
$$

$$
\min  _ {G} \quad \mathbb {E} _ {\mathbf {z} \sim P _ {\mathbf {z}}} [ \log (1 - D (G (\mathbf {z}))) ]. \tag {2}
$$

$$
V (G, D) = \mathbb {E} _ {\mathbf {x} \sim P _ {\mathrm {d a t a}}} [ \log D (\mathbf {x}) ] + \mathbb {E} _ {\mathbf {z} \sim P _ {\mathbf {z}}} [ \log (1 - D (G (\mathbf {z}))) ]. \tag {3}
$$

$$
\mathcal {L} _ {A E} = \left\| \mathbf {x} ^ {\prime} - \mathbf {x} \right\| _ {2} ^ {2}, \tag {4}
$$

$$
G _ {D e c} (\mathbf {z}) = D e c (G (\mathbf {z})),
$$

$$
\hat {\mathbf {x}}, \hat {\mathbf {y}} = G _ {D e c} (\mathbf {z}, s), \mathbf {z} \sim P _ {\mathbf {z}} (\mathbf {z}), \tag {5}
$$

$$
\min  _ {G _ {D e c}} \max  _ {D _ {1}, D _ {2}} V \left(G _ {D e c}, D _ {1}, D _ {2}\right) = V _ {1} \left(G _ {D e c}, D _ {1}\right) + \lambda V _ {2} \left(G _ {D e c}, D _ {2}\right), \tag {6}
$$

$$
\begin{array}{l} V _ {1} (G _ {D e c}, D _ {1}) \\ = \mathbb {E} _ {s \sim P _ {\text {d a t a}} (s), (\mathbf {x}, y) \sim P _ {\text {d a t a}} (\mathbf {x}, y | s)} [ \log D _ {1} (\mathbf {x}, y, s) ] (7) \\ + \mathbb {E} _ {\hat {s} \sim P _ {G} (s), (\hat {x}, \hat {y}) \sim P _ {G} (x, y | s)} [ \log (1 - D _ {1} (\hat {x}, \hat {y}, \hat {s})) ], \\ V _ {2} \left(G _ {D e c}, D _ {2}\right) = \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {y}) \sim P _ {G} (\mathbf {x}, y | s = 1)} \left[ \log D _ {2} (\hat {\mathbf {x}}, \hat {y}) \right] (8) \\ + \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {y}) \sim P _ {G} (\mathbf {x}, y | s = 0)} [ \log (1 - D _ {2} (\hat {\mathbf {x}}, \hat {y})) ], (6) \\ \end{array}
$$

$$
\nabla_ {\theta_ {a e}} \frac {1}{m} \sum_ {i = 1} ^ {m} | | \mathbf {x} ^ {\prime} - \mathbf {x} | | _ {2} ^ {2}
$$

$$
\nabla_ {\theta_ {d _ {1}}} \frac {1}{m} \sum_ {i = 1} ^ {m} \left[ \log D _ {1} (\mathbf {x}, y, s) + \log (1 - D _ {1} (\hat {\mathbf {x}}, \hat {y}, \hat {s})) \right]
$$

$$
\nabla_ {\theta_ {g}} \frac {1}{m} \sum_ {i = 1} ^ {m} \log (1 - D _ {1} (\hat {\mathbf {x}}, \hat {y}, \hat {s}))
$$

$$
\nabla_ {\theta_ {d _ {2}}} \frac {1}{2 m} \sum_ {i = 1} ^ {2 m} \left[ \log D _ {2} (\hat {\mathbf {x}}, \hat {y}) + \log (1 - D _ {2} (\hat {\mathbf {x}}, \hat {y})) \right]
$$

$$
\nabla_ {\theta_ {g}} \frac {1}{2 m} \sum_ {i = 1} ^ {2 m} \left[ \log D _ {2} (\hat {\mathbf {x}}, \hat {y}) + \log (1 - D _ {2} (\hat {\mathbf {x}}, \hat {y})) \right]
$$

$$
D _ {1} ^ {*} (\mathbf {x}, y, s) = \frac {P _ {\mathrm {d a t a}} (\mathbf {x} , y , s)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y , s) + P _ {G} (\mathbf {x} , y , s)},
$$

$$
D _ {2} ^ {*} (\mathbf {x}, y) = \frac {P _ {G} (\mathbf {x} , y | s = 1)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)}
$$

$$
\begin{array}{l} V (G _ {D e c}, D _ {1}, D _ {2}) \\ = \int_ {x} \int_ {y} P _ {\text {d a t a}} (\mathbf {x}, y, s) \log D _ {1} (\mathbf {x}, y, s) + \int_ {x} \int_ {y} P _ {G} (\mathbf {x}, y, s) \log (1 - D _ {1} (\mathbf {x}, y, s)) \\ + \lambda \int_ {x} \int_ {y} P _ {G} (\mathbf {x}, y | s = 1) \log \left(D _ {2} (\mathbf {x}, y)\right) + \lambda \int_ {x} \int_ {y} P _ {G} (\mathbf {x}, y | s = 0) \log \left(1 - D _ {2} (\mathbf {x}, y)\right) \tag {9} \\ \end{array}
$$

$$
\begin{array}{l} C (G _ {D e c}) \\ = \max  _ {D _ {1}, D _ {2}} V (G _ {D e c}, D _ {1}, D _ {2}) \\ = \mathbb {E} _ {(\mathbf {x}, y, s) \sim P _ {\mathrm {d a t a}} (\mathbf {x}, y, s)} [ \log \frac {P _ {\mathrm {d a t a}} (\mathbf {x} , y , s)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y , s) + P _ {G} (\mathbf {x} , y , s)} ] \\ + \mathbb {E} _ {(\mathbf {x}, y, s) \sim P _ {G} (\mathbf {x}, y, s)} [ \log \frac {P _ {G} (\mathbf {x} , y , s)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y , s) + P _ {G} (\mathbf {x} , y , s)} ] \tag {10} \\ + \lambda \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {G} (\mathbf {x}, y | s = 1)} [ \log \frac {P _ {G} (\mathbf {x} , y | s = 1)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ + \lambda \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {G} (\mathbf {x}, y | s = 0)} [ \log \frac {P _ {G} (\mathbf {x} , y | s = 0)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ = - (2 + \lambda) \log 4 + 2 \cdot J S D \left(P _ {\text {d a t a}} (\mathbf {x}, y, s) | | P _ {G} (\mathbf {x}, y, s)\right) \\ + 2 \lambda \cdot J S D (P _ {G} (\mathbf {x}, y | s = 1) | | P _ {G} (\mathbf {x}, y | s = 0)), \\ \end{array}
$$

$$
\begin{array}{l} \min  _ {G _ {D e c}} \max  _ {D} V (G _ {D e c}, D) = \mathbb {E} _ {\mathbf {x}, y \sim P _ {\mathrm {d a t a}} (\mathbf {x}, y)} [ \log D _ {1} (\mathbf {x}, y) ] \\ + \mathbb {E} _ {\hat {\mathbf {x}}, \hat {y} \sim P _ {G} (\mathbf {x}, y)} [ \log (1 - D _ {1} (\hat {\mathbf {x}}, \hat {y})) ], \\ \end{array}
$$

$$
\min  _ {G _ {D e c}} \max  _ {D _ {1}, D _ {2}} V \left(G _ {D e c}, D _ {1}, D _ {2}\right) = V _ {1} \left(G _ {D e c}, D _ {1}\right) + V _ {2} \left(G _ {D e c}, D _ {2}\right) \tag {11}
$$

$$
\begin{array}{l} V _ {1} \left(G _ {D e c}, D _ {1}\right) = \mathbb {E} _ {\left(\mathbf {x}, y\right) \sim P _ {\mathrm {d a t a}} \left(\mathbf {x}, y \mid s = 1\right)} \left[ \log D _ {1} (\mathbf {x}, y) \right] \\ + \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {\text {d a t a}} (\mathbf {x}, y | s = 0)} [ \log D _ {1} (\mathbf {x}, y) ] \\ + \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {\mathbf {y}}) \sim P _ {G} (\mathbf {x}, \mathbf {y} | s = 1)} [ \log (1 - D _ {1} (\hat {\mathbf {x}}, \hat {\mathbf {y}})) ] \\ + \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {\mathbf {y}}) \sim P _ {G} (\mathbf {x}, \mathbf {y} | s = 0)} [ \log (1 - D _ {1} (\hat {\mathbf {x}}, \hat {\mathbf {y}})) ], \\ \end{array}
$$

$$
\begin{array}{l} V _ {2} \left(G _ {D e c}, D _ {2}\right) = \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {\mathbf {y}}) \sim P _ {G} (\mathbf {x}, \mathbf {y} | s = 1)} \left[ \log D _ {2} (\hat {\mathbf {x}}, \hat {\mathbf {y}}) \right] \\ + \mathbb {E} _ {(\hat {\mathbf {x}}, \hat {\mathbf {y}}) \sim P _ {G} (\mathbf {x}, \mathbf {y} | s = 0)} [ \log (1 - D _ {2} (\hat {\mathbf {x}}, \hat {\mathbf {y}})) ], \\ \end{array}
$$

$$
D _ {1} ^ {*} (\mathbf {x}, y) = \frac {P _ {\text {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\text {d a t a}} (\mathbf {x} , y | s = 0)}{P _ {\text {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\text {d a t a}} (\mathbf {x} , y | s = 0) + P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)},
$$

$$
D _ {2} ^ {*} (\mathbf {x}, y) = \frac {P _ {G} (\mathbf {x} , y | s = 1)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)}
$$

$$
\begin{array}{l} C ^ {\prime} (G) = \max  _ {D _ {1}, D _ {2}} V (G _ {D e c}, D _ {1}, D _ {2}) \\ = \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {\mathrm {d a t a}} (\mathbf {x}, y | s = 1)} [ \log \frac {P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 0)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 0) + P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ + \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {\mathrm {d a t a}} (\mathbf {x}, y | s = 0)} [ \log \frac {P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 0)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 0) + P _ {G} (\mathbf {x} , \hat {y} | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ + \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {G} (\mathbf {x}, y | s = 1)} [ \log \frac {P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)}{P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 1) + P _ {\mathrm {d a t a}} (\mathbf {x} , y | s = 0) + P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ + \mathbb {E} _ {(\mathbf {x}, \mathcal {Y}) \sim P _ {G} (\mathbf {x}, \mathcal {Y} | s = 0)} [ \log \frac {P _ {G} (\mathbf {x} , \mathcal {Y} | s = 1) + P _ {G} (\mathbf {x} , \mathcal {Y} | s = 0)}{P _ {\text {d a t a}} (\mathbf {x} , \mathcal {Y} | s = 1) + P _ {\text {d a t a}} (\mathbf {x} , \mathcal {Y} | s = 0) + P _ {G} (\mathbf {x} , \mathcal {Y} | s = 1) + P _ {G} (\mathbf {x} , \mathcal {Y} | s = 0)} ] \\ + \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {G} (\mathbf {x}, y | s = 1)} [ \log \frac {P _ {G} (\mathbf {x} , y | s = 1)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ + \mathbb {E} _ {(\mathbf {x}, y) \sim P _ {G} (\mathbf {x}, y | s = 0)} [ \log \frac {P _ {G} (\mathbf {x} , y | s = 0)}{P _ {G} (\mathbf {x} , y | s = 1) + P _ {G} (\mathbf {x} , y | s = 0)} ] \\ \end{array}
$$

$$
\begin{array}{l} C ^ {\prime} (G) = - 4 \log 4 + 2 \cdot J S D \left(P _ {G} (\mathbf {x}, y | s = 1) \mid \mid P _ {G} (\mathbf {x}, y | s = 0)\right) \\ + 2 J S D \left(P _ {\text {d a t a}} (\mathbf {x}, y | s = 1) + P _ {\text {d a t a}} (\mathbf {x}, y | s = 0) \| P _ {G} (\mathbf {x}, y | s = 1) + P _ {G} (\mathbf {x}, y | s = 0)\right). \tag {12} \\ \end{array}
$$
