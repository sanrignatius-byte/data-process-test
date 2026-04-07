$$
\| L (\mathcal {A} _ {S}, \cdot) - L (\mathcal {A} _ {S ^ {i}}, \cdot) \| _ {\infty} \leq \beta_ {N},
$$

$$
\operatorname * {P r} _ {S, S ^ {i} \in \mathcal {D} ^ {N}, X \sim \Im} \left[ \mathrm {I} \left[ \mathcal {A} _ {S} (X) \geq 0 \right] \neq \mathrm {I} \left[ \mathcal {A} _ {S ^ {i}} (X) \geq 0 \right] \right] \leq \beta_ {N},
$$

$$
| f (x) - f ^ {\prime} (x) | \leq \tau \cdot \left| L (f, s) - L (f ^ {\prime}, s) \right|,
$$

$$
| \mathcal {A} _ {S} (x) - \mathcal {A} _ {S ^ {i}} (x) | \leq \tau \cdot | L (\mathcal {A} _ {S}, s) - L (\mathcal {A} _ {S ^ {i}}, s) | \leq \tau \beta_ {N}.
$$

$$
\operatorname {I} \left[ \mathcal {A} _ {S} (x) \geq 0 \right] = \operatorname {I} \left[ \mathcal {A} _ {S ^ {i}} (x) \geq 0 \right].
$$

$$
\min  _ {f \in \mathcal {F}} \frac {1}{N} \sum_ {i \in [ N ]} L (f, s _ {i}) \quad s. t.
$$

$$
\Omega_ {S} (f) \leq 0.
$$

$$
\min  _ {f \in \mathcal {F}} \frac {1}{N} \sum_ {i \in [ N ]} L (f, s _ {i}) + \lambda \| f \| _ {k} ^ {2} \quad s. t. \tag {Stable-Fair}
$$

$$
\Omega (f) \leq 0.
$$

$$
f (x) = \left\langle \sum_ {i \in [ N ]} \alpha_ {i} k \left(x _ {i}, \cdot\right), k (x, \cdot) \right\rangle = \sum_ {i \in [ N ]} \alpha_ {i} k \left(x _ {i}, x\right). \tag {1}
$$

$$
f (x) \stackrel {(1)} {=} \sum_ {i \in [ N ]} \alpha_ {i} k (x _ {i}, x) = \sum_ {i \in [ N ]} \alpha_ {i} \langle x _ {i}, x \rangle = \langle \sum_ {i \in [ N ]} \alpha_ {i} x _ {i}, x \rangle = \langle \beta , x \rangle ,
$$

$$
\min  _ {f \in \mathcal {F}} \frac {1}{N} \sum_ {i \in [ N ]} L (f, s _ {i}) + \mu \cdot \Omega (f). \quad (\text {R e g F a i r})
$$

$$
\min _ {f \in \mathcal {F}} \frac {1}{N} \sum_ {i \in [ N ]} L (f, s _ {i}) + \mu \cdot \Omega (f) + \lambda \| f \| _ {k} ^ {2}.
$$

$$
S ^ {i} := \left\{s _ {1}, \ldots , s _ {i - 1}, s _ {i} ^ {\prime}, s _ {i + 1}, \ldots , s _ {N} \right\}.
$$

$$
\left| L (f (x), y) - L (f (x ^ {\prime}), y) \right| \leq \sigma \left| f (x) - f (x ^ {\prime}) \right|.
$$

$$
\mathbb {E} _ {S \sim \Im^ {N}} \left[ R (\mathcal {A} _ {S}) \right] - \mathbb {E} _ {s \sim \Im} \left[ L (f ^ {\star}, s) \right] \leq \frac {\sigma^ {2} \kappa^ {2}}{\lambda N} + \lambda B ^ {2}.
$$

$$
R (\mathcal {A} _ {S}) \leq E (\mathcal {A} _ {S}) + 8 \sqrt {\left(\frac {2 \sigma^ {2} \kappa^ {2}}{\lambda N} + \frac {1}{N}\right) \cdot \ln (8 / \delta)}.
$$

$$
G = \sup_{f = \alpha^{\top}\phi (\cdot)\in \mathcal{F}:\Omega (f)\leq 0}\sup_{s\in \mathcal{D}}\| \nabla_{\alpha}L(f,s)\|_{2}.
$$

$$
\mathbb {E} _ {S \sim \Im^ {N}} \left[ R (\mathcal {A} _ {S}) \right] - \mathbb {E} _ {s \sim \Im} \left[ L (f ^ {\star}, s) \right] \leq \frac {G ^ {2}}{\lambda N} + \lambda B ^ {2}.
$$

$$
G = \| \nabla_ {\alpha} L (f, s) \| _ {2} = \| \nabla_ {f} L (f, s) \cdot \phi (x) \| _ {2}.
$$

$$
R (\mathcal {A} _ {S}) \leq E (\mathcal {A} _ {S}) + 8 \sqrt {\left(\frac {2 G ^ {2}}{\lambda N} + \frac {1}{N}\right) \cdot \ln (8 / \delta)}.
$$

$$
f (\cdot) = \sum_ {i \in [ N ]} \alpha_ {i} k (x _ {i}, \cdot)
$$

$$
L (f, s) = (1 - y f (x)) _ {+}
$$

$$
\min  _ {\alpha \in \mathbb {R} ^ {N}} \sum_ {i \in [ N ]} \left(1 - y _ {i} \sum_ {j \in [ N ]} \alpha_ {j} k \left(x _ {j}, x _ {i}\right)\right) _ {+} + \lambda \| \sum_ {i, j \in [ N ]} \alpha_ {i} \alpha_ {j} k \left(x _ {i}, x _ {j}\right) \| _ {k} ^ {2} s. t. \tag {SVM}
$$

$$
\Omega (f) \leq 0.
$$

$$
L (f, s) = (f (x) - y) ^ {2}.
$$

$$
\min  _ {\alpha \in \mathbb {R} ^ {N}} \sum_ {i \in [ N ]} \left(y _ {i} - \sum_ {j \in [ N ]} \alpha_ {j} k \left(x _ {j}, x _ {i}\right)\right) ^ {2} + \lambda \| \sum_ {i, j \in [ N ]} \alpha_ {i} \alpha_ {j} k \left(x _ {i}, x _ {j}\right) \| _ {k} ^ {2} \quad s. t. \tag {LS}
$$

$$
\Omega (f) \leq 0.
$$

$$
\begin{array}{l} \left| \left(f (x) - y\right) ^ {2} - \left(f \left(x ^ {\prime}\right) - y\right) ^ {2} \right| \\ = \left| (f (x) - f \left(x ^ {\prime}\right)) \cdot (f (x) + f \left(x ^ {\prime}\right) - 2 y) \right| \\ \leq (| f (x) | + | f \left(x ^ {\prime}\right) | + 2) \cdot \left| (f (x) - f \left(x ^ {\prime}\right)) \right| \\ \leq (2 B + 2) \left| \left(f (x) - f \left(x ^ {\prime}\right)\right) \right|. \\ \end{array}
$$

$$
L (f, s) = \ln (1 + e ^ {- y f (x)}).
$$

$$
\begin{array}{l} \min  _ {\alpha \in \mathbb {R} ^ {N}} \sum_ {i \in [ N ]} \ln \left(1 + y _ {i} \cdot e ^ {- \sum_ {j \in [ N ]} \alpha_ {j} k \left(x _ {j}, x _ {i}\right)}\right) + \lambda \| \sum_ {i, j \in [ N ]} \alpha_ {i} \alpha_ {j} k \left(x _ {i}, x _ {j}\right) \| _ {k} ^ {2} \quad s. t. \tag {LR} \\ \Omega (f) \leq 0. \\ \end{array}
$$

$$
\| g - g ^ {i} \| _ {k} ^ {2} \leq \frac {\sigma}{2 \lambda N} \left(| g (x _ {i}) - g ^ {i} (x _ {i}) | + | g (x _ {i} ^ {\prime}) - g ^ {i} (x _ {i} ^ {\prime}) |\right).
$$

$$
d _ {F} (f, f ^ {\prime}) = F (f) - F \left(f ^ {\prime}\right) - \langle f - f ^ {\prime}, \nabla F \left(f ^ {\prime}\right) \rangle , \forall f, f ^ {\prime} \in \mathcal {F}.
$$

$$
R (f) := \frac {1}{N} \sum_ {i \in [ N ]} L (f, s _ {i}) + \lambda \| f \| _ {k} ^ {2}, \forall f \in \mathcal {F}.
$$

$$
R ^ {i} (f) := \frac {1}{N} \left(\sum_ {j \neq i} L (f, s _ {j}) + L (f, s _ {i} ^ {\prime})\right) + \lambda \| f \| _ {k} ^ {2}, \forall f \in \mathcal {F}.
$$

$$
d _ {R} \left(g ^ {i}, g\right) = R \left(g ^ {i}\right) - R (g) - \langle g ^ {i} - g, \nabla R (g) \rangle \leq R \left(g ^ {i}\right) - R (g), \tag {2}
$$

$$
d _ {R ^ {i}} (g, g ^ {i}) = R ^ {i} (g) - R ^ {i} (g ^ {i}) - \langle g - g ^ {i}, \nabla R ^ {i} (g ^ {i}) \rangle \leq R ^ {i} (g) - R ^ {i} (g ^ {i}). \tag {3}
$$

$$
\begin{array}{l} d _ {R} (g ^ {i}, g) + d _ {R ^ {i}} (g, g ^ {i}) \\ \leq R \left(g ^ {i}\right) - R (g) + R ^ {i} (g) - R ^ {i} \left(g ^ {i}\right) \tag {4} \\ = \frac {1}{N} \left(L \left(g ^ {i}, s _ {i}\right) - L \left(g, s _ {i}\right) + L \left(g, s _ {i} ^ {\prime}\right) - L \left(g ^ {i}, s _ {i} ^ {\prime}\right)\right). \\ \end{array}
$$

$$
\begin{array}{l} 2 \lambda \| g - g ^ {i} \| _ {k} ^ {2} \\ = \lambda d _ {\left\| \cdot \right\| _ {k} ^ {2}} \left(g, g ^ {i}\right) + \lambda d _ {\left\| \cdot \right\| _ {k} ^ {2}} \left(g ^ {i}, g\right) \quad \text {(D e f n . o f} \| \cdot \| _ {k} ^ {2}) \\ = d _ {R ^ {i}} (g, g ^ {i}) - d _ {\sum_ {j \neq i} L (\cdot , s _ {j})} (g, g ^ {i}) + d _ {R} (g ^ {i}, g) - d _ {\sum_ {i} L (\cdot , s _ {i})} (g ^ {i}, g) \quad \left(d _ {A + B} = d _ {A} + d _ {B}\right) \\ \leq d _ {R ^ {i}} \left(g, g ^ {i}\right) + d _ {R} \left(g ^ {i}, g\right) \quad \text {(n o n n e g a t i v i t y o f} d _ {F}) \tag {5} \\ \leq \frac {1}{N} \left(L \left(g ^ {i}, s _ {i}\right) - L (g, s _ {i}) + L \left(g, s _ {i} ^ {\prime}\right) - L \left(g ^ {i}, s _ {i} ^ {\prime}\right)\right) \\ \leq \frac {\sigma}{N} \left(| g (x _ {i}) - g ^ {i} (x _ {i}) | + | g (x _ {i} ^ {\prime}) - g ^ {i} (x _ {i} ^ {\prime}) |\right). \quad \left(L (\cdot , \cdot) \text {i s} \sigma \text {- a d m i s s i b l e}\right) \\ \end{array}
$$

$$
\left| g \left(x _ {i}\right) - g ^ {i} \left(x _ {i}\right) \right| \leq \left\| g - g ^ {i} \right\| _ {k} \sqrt {k \left(x _ {i} , x _ {i}\right)} \leq \kappa \| g - g ^ {i} \| _ {k},
$$

$$
| g (x _ {i} ^ {\prime}) - g ^ {i} (x _ {i} ^ {\prime}) | \leq \| g - g ^ {i} \| _ {k} \sqrt {k (x _ {i} ^ {\prime} , x _ {i} ^ {\prime})} \leq \kappa \| g - g ^ {i} \| _ {k}.
$$

$$
\left| L (g, s) - L \left(g ^ {i}, s\right) \right| \leq \sigma \left| g (x) - g ^ {i} (x) \right| \leq \frac {\sigma^ {2} \kappa^ {2}}{\lambda N}.
$$

$$
\mathbb {E} _ {S \sim \Im^ {N}} [ F (g) - F (h) ] \leq \frac {\sigma^ {2} \kappa^ {2}}{\lambda N}. \tag {6}
$$

$$
\begin{array}{l} \mathbb {E} _ {S \sim \Im^ {N}} \left[ R \left(\mathcal {A} _ {S}\right) \right] - \mathbb {E} _ {s \sim \Im} \left[ L \left(f ^ {\star}, s\right) \right] \\ = \mathbb {E} _ {S \sim \Im^ {N}} \left[ R (\mathcal {A} _ {S}) - \frac {1}{N} \sum_ {i \in [ N ]} L (f ^ {\star}, s _ {i}) \right] \\ = \mathbb {E} _ {S \sim \mathfrak {I} ^ {N}} \left[ F (g) - \lambda \| g \| _ {k} ^ {2} - F \left(f ^ {\star}\right) + \lambda \| f ^ {\star} \| _ {k} ^ {2} \right] \quad (\text {D e f n s . o f} g \text {a n d} F (\cdot)) \\ \leq \mathbb {E} _ {S \sim \mathfrak {I} ^ {N}} [ F (g) - F (f ^ {\star}) ] + \lambda \| f ^ {\star} \| _ {k} ^ {2} \quad (\| g \| _ {k} ^ {2} \geq 0) \\ \leq \frac {\sigma^ {2} \kappa^ {2}}{\lambda N} + \lambda B ^ {2} \quad \text {(I n e q . (6) a n d \| f ^ {\star} \| _ {k} \leq B)}. \\ \end{array}
$$

$$
| T | \cdot \operatorname * {P r} _ {S, S ^ {\prime} \sim \mathfrak {S} ^ {N}, X \sim T, \mathcal {A}} \left[ \mathrm {I} \left[ \mathcal {A} _ {S} (X) \geq 0 \right] \neq \mathrm {I} \left[ \mathcal {A} _ {S ^ {\prime}} (X) \geq 0 \right] \right].
$$

$$
\operatorname {s t a b} _ {T, n} (\mathcal {A}) := \frac {1}{n (n - 1)} \sum_ {i, j \in [ n ]: i \neq j} \sum_ {s = (x, z, y) \in T} \tag {7}
$$

$$
\left| \operatorname {I} \left[ \mathcal {A} _ {S _ {i}} (x) \geq 0 \right] - \operatorname {I} \left[ \mathcal {A} _ {S _ {j}} (x) \geq 0 \right] \right|.
$$

$$
\gamma (f) :=
$$

$$
\min  \left\{\frac {\Pr_ {D} [ f = 1 \mid Z = 0 ]}{\Pr_ {D} [ f = 1 \mid Z = 1 ]}, \frac {\Pr_ {D} [ f = 1 \mid Z = 1 ]}{\Pr_ {D} [ f = 1 \mid Z = 0 ]} \right\}. \tag {8}
$$

$$
2 \lambda \| v - v ^ {i} \| _ {2} ^ {2} \leq \frac {1}{N} \left(L \left(g ^ {i}, s _ {i}\right) - L \left(g, s _ {i}\right) + L \left(g, s _ {i} ^ {\prime}\right) - L \left(g ^ {i}, s _ {i} ^ {\prime}\right)\right). \tag {9}
$$

$$
\begin{array}{l} L (f, s) - L \left(f ^ {\prime}, s\right) \leq \left\langle \nabla_ {\alpha} L (f, s), \alpha - \alpha^ {\prime} \right\rangle \quad (\text {C o n v e x i t y} L (\cdot , s)) \\ \leq \left\| \nabla_ {\alpha} L (\alpha , s) \right\| _ {2} \cdot \left\| \alpha - \alpha^ {\prime} \right\| _ {2} \tag {10} \\ \leq G \| \alpha - \alpha^ {\prime} \| _ {2} \quad (\text {D e f n . o f} G). \\ \end{array}
$$

$$
\begin{array}{l} \left\| v - v ^ {i} \right\| _ {2} ^ {2} \leq \frac {1}{2 \lambda N} \left(L \left(g ^ {i}, s _ {i}\right) - L \left(g, s _ {i}\right) + L \left(g, s _ {i} ^ {\prime}\right) - L \left(g ^ {i}, s _ {i} ^ {\prime}\right)\right) \quad (\text {I n e q .} (9)) \\ \leq \frac {1}{2 \lambda N} \left(G \| v - v ^ {i} \| _ {2} + G \| v - v ^ {i} \| _ {2}\right) \quad \text {(I n e q . (1 0))} \\ = \frac {G}{\lambda N} \| v - v ^ {i} \| _ {2}. \\ \end{array}
$$

$$
L (g, s) - L (g ^ {i}, s) \leq G \| v - v ^ {i} \| _ {2} \leq \frac {G ^ {2}}{\lambda N}.
$$

$$
\begin{array}{l} \left| L (f (x), y) - L (f \left(x ^ {\prime}\right), y) \right| \\ = \left| \operatorname {I} [ f (x) \neq y ] - \operatorname {I} [ f (x ^ {\prime}) \neq y ] \right| \\ \end{array}
$$

$$
= \mathrm {I} [ f (x) \neq f \left(x ^ {\prime}\right) ] = \frac {1}{2} | f (x) - f \left(x ^ {\prime}\right) |.
$$

$$
\begin{array}{l} \left| L (f (x), y) - L \left(f \left(x ^ {\prime}\right), y\right) \right| \\ = \left| (1 - y f (x)) _ {+} - (1 - y f \left(x ^ {\prime}\right)) _ {+} \right| \\ \leq \left| y f (x) - y f \left(x ^ {\prime}\right) \right| \\ = \left| f (x) - f \left(x ^ {\prime}\right) \right|, \\ \end{array}
$$

$$
\begin{array}{l} \left| L (f (x), y) - L \left(f \left(x ^ {\prime}\right), y\right) \right| \\ = \left| (f (x) - y) ^ {2} - \left(f \left(x ^ {\prime}\right) - y\right) ^ {2} \right| \\ = \left| (f (x) + f \left(x ^ {\prime}\right) - 2 y) \left(f (x) - f \left(x ^ {\prime}\right)\right) \right| \\ \leq 4 \left| f (x) - f \left(x ^ {\prime}\right) \right|, \\ \end{array}
$$

$$
\begin{array}{l} \left| \nabla_ {f (x)} \ln \left(1 + e ^ {- y f (x)}\right) \right| \\ = \left| \frac {- y e ^ {- y f (x)}}{1 + e ^ {- y f (x)}} \right| = \left| \frac {e ^ {- y f (x)}}{1 + e ^ {- y f (x)}} \right| \leq 1. \\ \end{array}
$$
