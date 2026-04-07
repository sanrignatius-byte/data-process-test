$$
\mathbf {z} \sim p (\mathbf {z}); \qquad \mathbf {x} \sim p _ {\theta} (\mathbf {x} | \mathbf {z}, \mathbf {s})
$$

$$
\begin{array}{l} \sum_ {n = 1} ^ {N} \log p (\mathbf {x} _ {n} | \mathbf {s} _ {n}) \geq \sum_ {n = 1} ^ {N} \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {n} | \mathbf {x} _ {n}, \mathbf {s} _ {n})} [ \log p _ {\theta} (\mathbf {x} _ {n} | \mathbf {z} _ {n}, \mathbf {s} _ {n}) ] - K L (q _ {\phi} (\mathbf {z} _ {n} | \mathbf {x} _ {n}, \mathbf {s} _ {n}) | | p (\mathbf {z})) \tag {1} \\ = \mathcal {F} (\phi , \theta ; \mathbf {x} _ {n}, \mathbf {s} _ {n}) \\ \end{array}
$$

$$
\mathbf {y}, \mathbf {z} _ {2} \sim \operatorname {C a t} (\mathbf {y}) p (\mathbf {z} _ {2}); \qquad \mathbf {z} _ {1} \sim p _ {\theta} (\mathbf {z} _ {1} | \mathbf {z} _ {2}, \mathbf {y}); \qquad \mathbf {x} \sim p _ {\theta} (\mathbf {x} | \mathbf {z} _ {1}, \mathbf {s})
$$

$$
\begin{array}{l} \sum_ {n = 1} ^ {N} \log p (\mathbf {x} _ {n} | \mathbf {s} _ {n}) \geq \sum_ {n = 1} ^ {N} \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 n}, \mathbf {z} _ {2 n}, \mathbf {y} _ {n} | \mathbf {x} _ {n}, \mathbf {s} _ {n})} [ \log p (\mathbf {z} _ {2}) + \log p (\mathbf {y} _ {n}) + \log p _ {\theta} (\mathbf {z} _ {1 n} | \mathbf {z} _ {2 n}, \mathbf {y} _ {n}) + \\ + \log p _ {\theta} (\mathbf {x} _ {n} | \mathbf {z} _ {1 _ {n}}, \mathbf {s} _ {n}) - \log q _ {\phi} (\mathbf {z} _ {1 _ {n}}, \mathbf {z} _ {2 _ {n}}, \mathbf {y} _ {n} | \mathbf {x} _ {n}, \mathbf {s} _ {n}) ] \tag {2} \\ \end{array}
$$

$$
q _ {\phi} \left(\mathbf {z} _ {1 n} \mid \mathbf {x} _ {n}, \mathbf {s} _ {n}\right) = \mathcal {N} \left(\mathbf {z} _ {1 n} \mid \boldsymbol {\mu} _ {n} = f _ {\phi} \left(\mathbf {x} _ {n}, \mathbf {s} _ {n}\right), \boldsymbol {\sigma} _ {n} = e ^ {f _ {\phi} \left(\mathbf {x} _ {n}, \mathbf {s} _ {n}\right)}\right)
$$

$$
q _ {\phi} (\mathbf {y} _ {n} | \mathbf {z} _ {1 n}) = \operatorname {C a t} (\mathbf {y} _ {n} | \boldsymbol {\pi} _ {n} = \operatorname {s o f t m a x} (f _ {\phi} (\mathbf {z} _ {1 n})))
$$

$$
q _ {\phi} \left(\mathbf {z} _ {2 n} \mid \mathbf {z} _ {1 n}, \mathbf {y} _ {n}\right) = \mathcal {N} \left(\mathbf {z} _ {2 n} \mid \boldsymbol {\mu} _ {n} = f _ {\phi} \left(\mathbf {z} _ {1 n}, \mathbf {y} _ {n}\right), \boldsymbol {\sigma} _ {n} = e ^ {f _ {\phi} \left(\mathbf {z} _ {1 n}, \mathbf {y} _ {n}\right)}\right)
$$

$$
p _ {\theta} \left(\mathbf {z} _ {1 n} \mid \mathbf {z} _ {2 n}, \mathbf {y} _ {n}\right) = \mathcal {N} \left(\mathbf {z} _ {1 n} \mid \boldsymbol {\mu} _ {n} = f _ {\theta} \left(\mathbf {z} _ {2 n}, \mathbf {y} _ {n}\right), \boldsymbol {\sigma} _ {n} = e ^ {f _ {\theta} \left(\mathbf {z} _ {2 n}, \mathbf {y} _ {n}\right)}\right)
$$

$$
p _ {\theta} (\mathbf {x} _ {n} | \mathbf {z} _ {1 n}, \mathbf {s} _ {n}) = f _ {\theta} (\mathbf {z} _ {1 n}, \mathbf {s} _ {n})
$$

$$
\begin{array}{l} \sum_ {n = 1} ^ {N} \mathcal {L} _ {s} (\phi , \theta ; \mathbf {x} _ {n}, \mathbf {s} _ {n}, \mathbf {y} _ {n}) = \sum_ {n = 1} ^ {N _ {s}} \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 _ {n}} | \mathbf {x} _ {n}, \mathbf {s} _ {n})} [ - K L (q _ {\phi} (\mathbf {z} _ {2 _ {n}} | \mathbf {z} _ {1 _ {n}}, \mathbf {y} _ {n}) | | p (\mathbf {z} _ {2})) + \log p _ {\theta} (\mathbf {x} _ {n} | \mathbf {z} _ {1 _ {n}}, \mathbf {s} _ {n}) ] + \\ + \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 _ {n}} | \mathbf {x} _ {n}, \mathbf {s} _ {n}) q _ {\phi} (\mathbf {z} _ {2 _ {n}} | \mathbf {z} _ {1 _ {n}}, \mathbf {y} _ {n})} [ \log p _ {\theta} (\mathbf {z} _ {1} | \mathbf {z} _ {2 _ {n}}, \mathbf {y} _ {n}) - \log q _ {\phi} (\mathbf {z} _ {1 _ {n}} | \mathbf {x} _ {n} \mathbf {s} _ {n}) ] \tag {3} \\ \end{array}
$$

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} \mathcal {L} _ {u} (\phi , \theta ; \mathbf {x} _ {m}, \mathbf {s} _ {m}) = \sum_ {m = 1} ^ {M} \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 m} | \mathbf {x} _ {m}, \mathbf {s} _ {m})} [ - K L (q (\mathbf {y} _ {m} | \mathbf {z} _ {1 m}) | | p (\mathbf {y} _ {m})) + \log p _ {\theta} (\mathbf {x} _ {m} | \mathbf {z} _ {1 m}, \mathbf {s} _ {m}) ] + \\ + \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 m}, \mathbf {y} _ {m} | \mathbf {x} _ {m}, \mathbf {s} _ {m})} [ - K L (q _ {\phi} (\mathbf {z} _ {2 m} | \mathbf {z} _ {1 m}, \mathbf {y} _ {m}) | | p (\mathbf {z} _ {2})) ] + \\ + \mathbb {E} _ {q _ {\phi} (\mathbf {z} _ {1 m}, \mathbf {y} _ {m}, \mathbf {z} _ {2 m} | \mathbf {x} _ {m}, \mathbf {s} _ {m})} [ \log p _ {\theta} (\mathbf {z} _ {1 m} | \mathbf {z} _ {2 m}, \mathbf {y} _ {m}) - \log q _ {\phi} (\mathbf {z} _ {1 m} | \mathbf {x} _ {m}, \mathbf {s} _ {m}) ] \tag {4} \\ \end{array}
$$

$$
\begin{array}{l} \mathcal {F} _ {\mathrm {V A E}} \left(\phi , \theta ; \mathbf {x} _ {n}, \mathbf {x} _ {m}, \mathbf {s} _ {n}, \mathbf {s} _ {m}, \mathbf {y} _ {n}\right) = \sum_ {n = 1} ^ {N} \mathcal {L} _ {s} \left(\phi , \theta ; \mathbf {x} _ {n}, \mathbf {s} _ {n}, \mathbf {y} _ {n}\right) + \sum_ {m = 1} ^ {M} \mathcal {L} _ {u} \left(\phi , \theta ; \mathbf {x} _ {m}, \mathbf {s} _ {m}\right) + \\ + \alpha \sum_ {n = 1} ^ {N} \mathbb {E} _ {q \left(\mathbf {z} _ {1 n} \mid \mathbf {x} _ {n}, \mathbf {s} _ {n}\right)} [ - \log q _ {\phi} (\mathbf {y} _ {n} \mid \mathbf {z} _ {1 n}) ] \tag {5} \\ \end{array}
$$

$$
\left\| \frac {1}{N _ {0}} \sum_ {i = 1} ^ {N _ {0}} \psi \left(\mathbf {x} _ {i}\right) - \frac {1}{N _ {1}} \sum_ {i = 1} ^ {N _ {1}} \psi \left(\mathbf {x} _ {i} ^ {\prime}\right) \right\| ^ {2}. \tag {6}
$$

$$
\ell_ {\mathrm {M M D}} (\mathbf {X}, \mathbf {X} ^ {\prime}) = \frac {1}{N _ {0} ^ {2}} \sum_ {n = 1} ^ {N _ {0}} \sum_ {m = 1} ^ {N _ {0}} k \left(\mathbf {x} _ {n}, \mathbf {x} _ {m}\right) + \frac {1}{N _ {1} ^ {2}} \sum_ {n = 1} ^ {N _ {1}} \sum_ {m = 1} ^ {N _ {1}} k \left(\mathbf {x} _ {n} ^ {\prime}, \mathbf {x} _ {m} ^ {\prime}\right) - \frac {2}{N _ {0} N _ {1}} \sum_ {n = 1} ^ {N _ {0}} \sum_ {m = 1} ^ {N _ {1}} k \left(\mathbf {x} _ {n}, \mathbf {x} _ {m} ^ {\prime}\right). \tag {7}
$$

$$
\mathcal {F} _ {\mathrm {V F A E}} \left(\phi , \theta ; \mathbf {x} _ {n}, \mathbf {x} _ {m}, \mathbf {s} _ {n}, \mathbf {s} _ {m}, \mathbf {y} _ {n}\right) = \mathcal {F} _ {\mathrm {V A E}} \left(\phi , \theta ; \mathbf {x} _ {n}, \mathbf {x} _ {m}, \mathbf {s} _ {n}, \mathbf {s} _ {m}, \mathbf {y} _ {n}\right) - \beta \ell_ {\mathrm {M M D}} \left(\mathbf {Z} _ {1 \mathrm {s} = 0}, \mathbf {Z} _ {1 \mathrm {s} = 1}\right) \tag {8}
$$

$$
\ell_ {\mathrm {M M D}} (\mathbf {Z} _ {1 \mathbf {s} = 0}, \mathbf {Z} _ {1 \mathbf {s} = 1}) = \| \mathbb {E} _ {\tilde {p} (\mathbf {x} | \mathbf {s} = 0)} [ \mathbb {E} _ {q (\mathbf {z} _ {1} | \mathbf {x}, \mathbf {s} = 0)} [ \psi (\mathbf {z} _ {1}) ] ] - E _ {\tilde {p} (\mathbf {x} | \mathbf {s} = 1)} [ \mathbb {E} _ {q (\mathbf {z} _ {1} | \mathbf {x}, \mathbf {s} = 1)} [ \psi (\mathbf {z} _ {1}) ] ] \| ^ {2} (9)
$$

$$
\psi_ {\mathbf {W}} (\mathbf {x}) = \sqrt {\frac {2}{D}} \cos \left(\sqrt {\frac {2}{\gamma}} \mathbf {x} \mathbf {W} + \mathbf {b}\right). \tag {10}
$$

$$
\text {D i s c r i m i n a t i o n} = \left| \frac {\sum_ {n = 1} ^ {N} \mathbb {I} \left[ y _ {n} ^ {s = 0} \right]}{N _ {s = 0}} - \frac {\sum_ {n = 1} ^ {N} \mathbb {I} \left[ y _ {n} ^ {s = 1} \right]}{N _ {s = 1}} \right|
$$

$$
\text {D i s c r i m i n a t i o n} = \left| \frac {\sum_ {n = 1} ^ {N} p \left(y _ {n} ^ {s = 0}\right)}{N _ {s = 0}} - \frac {\sum_ {n = 1} ^ {N} p \left(y _ {n} ^ {s = 1}\right)}{N _ {s = 1}} \right|
$$

$$
\mathrm {P A D} (\epsilon) = 2 (1 - 2 \epsilon)
$$
