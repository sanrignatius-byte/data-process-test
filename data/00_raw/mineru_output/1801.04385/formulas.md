$$
\frac {d}{d x _ {p}} \mathbb {E} [ Y | X _ {p} = x _ {p} ] > 0 \quad \forall x _ {p}, \tag {1}
$$

$$
\frac {d}{d x _ {p}} \mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ] \leq 0 \quad \forall x _ {p}, x _ {c}. \tag {2}
$$

$$
\mathbb {E} [ Y | X _ {p} = x _ {p} ] = f _ {p} (\alpha + \beta x _ {p}), \tag {3}
$$

$$
\mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ] = f _ {p, c} \left(\alpha \left(x _ {c}\right) + \beta \left(x _ {c}\right) x _ {p}\right), \tag {4}
$$

$$
\mathbb {E} [ Y | X _ {p} = x _ {p} ] = \tag {5}
$$

$$
\int_ {X _ {c}} \mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ] \Pr (X _ {c} = x _ {c} | X _ {p} = x _ {p}) d x _ {c},
$$

$$
\int_ {X _ {c}} \left(\frac {d}{d x _ {p}} \mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ]\right) \operatorname * {P r} (X _ {c} = x _ {c} | X _ {p} = x _ {p}) d x _ {c} +
$$

$$
\int_ {X _ {c}} \mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ] \left(\frac {d}{d x _ {p}} \Pr (X _ {c} = x _ {c} | X _ {p} = x _ {p})\right) d x _ {c}. \tag {6}
$$

$$
\frac {d}{d x _ {p}} \Pr \left(X _ {c} = x _ {c} \mid X _ {p} = x _ {p}\right) \neq 0, \tag {7}
$$

$$
\mathbb {E} [ Y | X _ {p} = x _ {p}, X _ {c} = x _ {c} ] \neq \mathbb {E} [ Y | X _ {p} = x _ {p} ]. \tag {8}
$$

$$
\begin{array}{l} \int_ {X _ {c}} \mathbb {E} [ Y | X _ {p} = x _ {p} ] \left(\frac {d}{d x _ {p}} \Pr (X _ {c} = x _ {c} | X _ {p} = x _ {p})\right) d x _ {c} \tag {9} \\ = \mathbb {E} [ Y | X _ {p} = x _ {p} ] \frac {d}{d x _ {p}} \left(\int_ {X _ {c}} \operatorname * {P r} (X _ {c} = x _ {c} | X _ {p} = x _ {p}) d x _ {c}\right) = 0, \\ \end{array}
$$

$$
f (\alpha + \beta x) = \frac {1}{1 + e ^ {- (\alpha + \beta x)}}. \tag {10}
$$

$$
\frac {d}{d x _ {p}} \Pr (X _ {c} = a | X _ {p} = x _ {p}) \| _ {x _ {p} = a} = - \Pr (X _ {c} = a | X _ {p} = a). \tag {11}
$$

$$
\Pr \left(X _ {c} = x _ {c} \mid X _ {p} = a + 1\right) = \frac {\Pr \left(X _ {c} = x _ {c} \mid X _ {p} = a\right)}{1 - \Pr \left(X _ {c} = a \mid X _ {p} = a\right)}. \tag {12}
$$

$$
\begin{array}{l} \frac {d}{d x _ {p}} \Pr (X _ {c} = x _ {c} | X _ {p} = x _ {p}) \| _ {x _ {p} = a} = \\ \left(\frac {1}{1 - \Pr \left(X _ {c} = a \mid X _ {p} = a\right)} - 1\right) \Pr \left(X _ {c} = x _ {c} \mid X _ {p} = a\right). \tag {13} \\ \end{array}
$$
