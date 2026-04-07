$$
\Delta_ {D P} = | \mathbb {E} [ \bar {y} = 1 | a = 1 ] - \mathbb {E} [ \bar {y} = 1 | a = 0 ] | \tag {1}
$$

$$
L _ {\mathrm {V A E}} (p, q) = \mathbb {E} _ {q (z | x)} [ \log p (x | z) ] - D _ {K L} [ q (z | x) | | p (z) ],
$$

$$
q (z | x) = \mathcal {N} (z | \mu_ {q} (x), \Sigma_ {q} (x))
$$

$$
p (x | z) = \mathcal {N} (x | \mu_ {p} (z), \Sigma_ {p} (z))
$$

$$
L _ {\beta \mathrm {V A E}} (p, q) = \mathbb {E} _ {q (z | x)} [ \log p (x | z) ] - \beta D _ {K L} [ q (z | x) | | p (z) ].
$$

$$
L _ {\text {F a c t o r V A E}} (p, q) = L _ {\text {V A E}} (p, q) - \gamma D _ {K L} (q (z) | | \prod_ {j} q (z _ {j}))
$$

$$
\begin{array}{l} \mathbb {E} _ {p ^ {\text {d a t a}} (x)} [ D _ {K L} (q (z | x) | | p (z)) ] = \\ D _ {K L} (q (z | x) p ^ {\text {d a t a}} (x) \mid \mid q (z) p ^ {\text {d a t a}} (x)) \\ + D _ {K L} (q (z) | | \prod_ {j} q (z _ {j})) \\ + \sum_ {j} D _ {K L} [ q (z _ {j}) | | p (z _ {j}) ]. \\ \end{array}
$$

$$
\begin{array}{l} L _ {\mathrm {V A E}} (p, q) = \mathbb {E} _ {q (z, b | x, a)} [ \log p (x, a | z, b) ] \\ - D _ {K L} [ q (z, b | x, a) | | p (z, b) ]. \\ \end{array}
$$

$$
q (z, b | x) = q (z | x) q (b | x). \tag {2}
$$

$$
p (x, a \mid z, b) = p (x \mid z, b) p (a \mid b) \tag {3}
$$

$$
\begin{array}{l} L _ {\mathrm {F F V A E}} (p, q) = \mathbb {E} _ {q (z, b | x)} [ \log p (x | z, b) + \alpha \log p (a | b) ] \\ - \gamma D _ {K L} (q (z, b) | | q (z) \prod_ {j} q (b _ {j})) \\ - D _ {K L} [ q (z, b | x) | | p (z, b) ]. \tag {4} \\ \end{array}
$$

$$
\begin{array}{l} \log d (u = 1 | z, b) - \log d (u = 0 | z, b) \approx \\ \log q (z, b) - \log q (z) \prod_ {j} q \left(b _ {j}\right). \tag {5} \\ \end{array}
$$

$$
\begin{array}{l} D _ {K L} (q (z, b) | | q (z) \prod_ {j} q (b _ {j})) = \\ \mathbb {E} _ {q (z, b)} \left[ \log q (z, b) - \log q (z) \prod_ {j} q (b _ {j}) \right] \approx \\ \mathbb {E} _ {q (z, b)} [ \log d (u = 1 | z, b) - \log d (u = 0 | z, b) ]. \tag {6} \\ \end{array}
$$

$$
\begin{array}{l} L _ {\text {D i s c}} (d) = \mathbb {E} _ {z, b \sim q (z, b)} [ \log d (u = 1 | z, b) ] \\ + \mathbb {E} _ {z ^ {\prime}, b ^ {\prime} \sim q (z)} \prod_ {j} q (b _ {j}) [ \log (1 - d (u = 0 | z ^ {\prime}, b ^ {\prime})) ], \tag {7} \\ \end{array}
$$

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{H \left(v _ {k}\right)} \left(M I \left(z _ {j _ {k}}; v _ {k}\right) - \max  _ {j \neq j _ {k}} M I \left(z _ {j}; v _ {k}\right)\right) \tag {8}
$$
