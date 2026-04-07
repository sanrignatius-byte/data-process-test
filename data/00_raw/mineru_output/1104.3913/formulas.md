$$
D (M x, M y) \leq d (x, y). \tag {1}
$$

$$
\operatorname {o p t} (I) \stackrel {\text {d e f}} {=} \min  _ {\{\mu_ {x} \} _ {x \in V}} \mathbb {E} _ {x \sim V} \mathbb {E} _ {a \sim \mu_ {x}} L (x, a) \tag {2}
$$

$$
\text {s u b j e c t} \quad \forall x, y \in V,: \quad D \left(\mu_ {x}, \mu_ {y}\right) \leq d (x, y) \tag {3}
$$

$$
\forall x \in V: \quad \mu_ {x} \in \Delta (A) \tag {4}
$$

$$
D _ {\mathrm {t v}} (P, Q) = \frac {1}{2} \sum_ {a \in A} | P (a) - Q (a) |. \tag {5}
$$

$$
D _ {\infty} (P, Q) = \sup  _ {a \in A} \log \left(\max  \left\{\frac {P (a)}{Q (a)}, \frac {Q (a)}{P (a)} \right\}\right). \tag {6}
$$

$$
D _ {\mathrm {t v}} \left(\mu_ {S}, \mu_ {T}\right) \leq \epsilon . \tag {7}
$$

$$
\operatorname {b i a s} _ {D, d} (S, T) \stackrel {\text {d e f}} {=} \max  \mu_ {S} (0) - \mu_ {T} (0), \tag {8}
$$

$$
D _ {\mathrm {t v}} \left(\mu_ {S}, \mu_ {T}\right) = D _ {\mathrm {t v}} \left(\mu_ {S} ^ {\prime}, \mu_ {T} ^ {\prime}\right) = \mu_ {S} ^ {\prime} (0) - \mu_ {T} ^ {\prime} (0) \leq \operatorname {b i a s} _ {D, d} (S, T).
$$

$$
\begin{array}{l} \sigma_ {\mathrm {E M}} (S, T) \stackrel {{\mathrm {d e f}}} {{=}} \quad \min  \quad \sum_ {x, y \in V} h (x, y) \sigma (x, y) \\ \text {s u b j e c t} \quad \sum_ {y \in V} h (x, y) = S (x) \\ \sum_ {y \in V} h (y, x) = T (x) \\ h (x, y) \geq 0 \\ \end{array}
$$

$$
\begin{array}{l} d _ {\mathrm {E M}} (S, T) = \min  \sum_ {x, y \in V} h (x, y) d (x, y) \\ \text {s u b j e c t} \quad \sum_ {y \in V} h (x, y) = \sum_ {y \in V} h (y, x) + S (x) - T (x) \\ h (x, y) \geq 0 \\ \end{array}
$$

$$
\operatorname {b i a s} _ {D _ {\mathrm {t v}}, d} (S, T) \leq d _ {\mathrm {E M}} (S, T). \tag {9}
$$

$$
\operatorname {b i a s} _ {D _ {\mathrm {t v}}, d} (S, T) \geq d _ {\mathrm {E M}} (S, T). \tag {10}
$$

$$
\begin{array}{l} \operatorname {b i a s} (S, T) = \max  \sum_ {x \in V} S (x) \mu_ {x} (0) - \sum_ {x \in V} T (x) \mu_ {x} (0) \\ \text {s u b j e c t} \quad \mu_ {x} (0) - \mu_ {y} (0) \leq d (x, y) \\ \mu_ {x} (0) + \mu_ {x} (1) = 1 \\ \mu_ {x} (a) \geq 0 \\ \end{array}
$$

$$
D _ {\mathrm {t v}} \left(\mu_ {x}, \mu_ {y}\right) \leq d (x, y) \quad \Longleftrightarrow \quad \left| \mu_ {x} (0) - \mu_ {y} (0) \right| \leq d (x, y).
$$

$$
\begin{array}{l} \beta (S, T) \stackrel {{\text {d e f}}} {{=}} \quad \max  \quad \sum_ {x \in V} S (x) \mu_ {x} (0) - \sum_ {x \in V} T (x) \mu_ {x} (0) \\ \text {s u b j e c t} \quad \mu_ {x} (0) - \mu_ {y} (0) \leq d (x, y) \\ \mu_ {x} (0) \geq 0 \\ \end{array}
$$

$$
\operatorname {b i a s} (S, T) = \beta (S, T).
$$

$$
\begin{array}{l} \beta (S, T) = \quad \min  \quad \sum_ {x, y \in V} h (x, y) d (x, y) \\ \text {s u b j e c t} \quad \sum_ {y \in V} h (x, y) \geq \sum_ {y \in V} h (y, x) + S (x) - T (x) \\ h (x, y) \geq 0 \\ \end{array}
$$

$$
\operatorname {b i a s} _ {D _ {\infty}, d} (S, T) \leq \operatorname {b i a s} _ {D _ {\mathrm {t v}}, d} (S, T) \tag {11}
$$

$$
\operatorname {b i a s} _ {D _ {\infty}, d} (S, T) \leq d _ {\mathrm {E M}} (S, T) \tag {12}
$$

$$
\operatorname {b i a s} _ {D _ {\infty}, d} (S, T) = \quad \min  \quad \sum_ {x \in V} \epsilon_ {x}
$$

$$
\text {s u b j e c t} \quad \sum_ {y \in V} f (x, y) + \epsilon_ {x} \geq \sum_ {y \in V} f (y, x) e ^ {d (x, y)} + S (x) - T (x) \tag {13}
$$

$$
\sum_ {y \in V} g (x, y) + \epsilon_ {x} \geq \sum_ {y \in V} g (y, x) e ^ {d (x, y)} \tag {14}
$$

$$
f (x, y), g (x, y) \geq 0
$$

$$
L ^ {\prime} (y, a) = \sum_ {x \in S} \mu_ {x} (y) L (x, a) + L (y, a),
$$

$$
d _ {\mathrm {E M} + \mathrm {L}} (S, T) \stackrel {\text {d e f}} {=} \quad \min  \quad \underset {x \in S} {\mathbb {E}} \underset {y \sim \mu_ {x}} {\mathbb {E}} d (x, y) \tag {15}
$$

$$
\text {s u b j e c t} \quad D \left(\mu_ {x}, \mu_ {x ^ {\prime}}\right) \leq d \left(x, x ^ {\prime}\right) \quad \text {f o r a l l} \quad x, x ^ {\prime} \in S
$$

$$
D _ {\mathrm {t v}} \left(\mu_ {S}, U _ {T}\right) \leq \epsilon
$$

$$
\mu_ {x} \in \Delta (T) \quad \text {f o r a l l} \quad x \in S
$$

$$
M (x) = \left\{ \begin{array}{l l} v _ {x} & x \in T \\ \mathbb {E} _ {y \sim \mu_ {x}} v _ {y} & x \in S \end{array} . \right. \tag {16}
$$

$$
D _ {\mathrm {t v}} (M (S), M (T)) = D _ {\mathrm {t v}} \left(\underset {x \in S} {\mathbb {E}} \underset {y \sim \mu_ {x}} {\mathbb {E}} \nu_ {y}, \underset {x \in T} {\mathbb {E}} \nu_ {x}\right) \leq D _ {\mathrm {t v}} (\mu_ {S}, U _ {T}) \leq \epsilon .
$$

$$
D (M (x), M (y)) \leq D (\mu_ {x}, \mu_ {y}) \leq d (x, y).
$$

$$
\underset {x \in S} {\mathbb {E}} \max _ {y \in T} \left[ D _ {\mathrm {t v}} (M (x), M (y)) - d (x, y) \right] \leq d _ {\mathrm {E M + L}} (S, T).
$$

$$
\begin{array}{l} D _ {\mathrm {t v}} (M (x), M (y)) = D _ {\mathrm {t v}} \left(\underset {z \sim \mu_ {x}} {\mathbb {E}} M (z), M (y)\right) \\ \leq \mathbb {E} _ {z \sim \mu_ {x}} D _ {\mathrm {t v}} (M (z), M (y)) \quad (\text {b y}) \\ \leq \underset {z \sim \mu_ {x}} {\mathbb {E}} d (z, y) \quad (\text {P r o p o s i t i o n 4 . 1 s i n c e} z, y \in T) \\ \leq d (x, y) + \underset {z \sim \mu_ {x}} {\mathbb {E}} d (x, z) \quad (\text {b y}) \\ \end{array}
$$

$$
\operatorname {E} (x) \stackrel {\text {d e f}} {=} \left[ Z _ {x} ^ {- 1} e ^ {- d (x, y)} \right] _ {y \in V},
$$

$$
\underset {x \in V} {\mathbb {E}} \underset {y \sim \mathrm {E} (x)} {\mathbb {E}} d (x, y) = O (1).
$$

$$
\underset {x \in V} {\mathbb {E}} | B (x, 2 R) | \leq 2 ^ {k ^ {\prime}} \underset {x \in V} {\mathbb {E}} | B (x, R) |, \tag {17}
$$

$$
\underset {x \in V} {\mathbb {E}} | B (x, 1) | \leq \left(\frac {1}{\epsilon}\right) ^ {k ^ {\prime}} \underset {x \in V} {\mathbb {E}} | B (x, \epsilon) | = 2 ^ {O (k)}. \tag {18}
$$

$$
\begin{array}{l} \mathbb{E}_{x\in V}\mathbb{E}_{y\sim \mathrm{E}(x)}d(x,y)\leq 1 + \mathbb{E}_{x\in V}\int_{1}^{\infty}\frac{re^{-r}}{Z_{x}} |B(x,r)|\mathrm{d}r \\ \leq 1 + \mathbb {E} _ {x \in V} \int_ {1} ^ {\infty} r e ^ {- r} | B (x, r) | \mathrm {d} r \quad (\text {s i n c e} Z _ {x} \geq e ^ {- d (x, x)} = 1) \\ = 1 + \int_ {1} ^ {\infty} r e ^ {- r} \mathbb {E} _ {x \in V} | B (x, r) | \mathrm {d} r \\ \leq 1 + \int_ {1} ^ {\infty} r e ^ {- r} r ^ {k ^ {\prime}} \underset {x \in V} {\mathbb {E}} | B (x, 1) | \mathrm {d} r \quad \text {(u s i n g (1 8))} \\ \leq 1 + 2 ^ {O (k)} \int_ {0} ^ {\infty} r ^ {k ^ {\prime} + 1} e ^ {- r} d r \\ \leq 1 + 2 ^ {O (k)} \left(k ^ {\prime} + 2\right)! \\ \end{array}
$$

$$
\underset {x \in V} {\mathbb {E}} \underset {y \sim \operatorname {E} (x)} {\mathbb {E}} d (x, y) \leq 2 ^ {O (k)} (k ^ {\prime} + 2)! \leq O (1).
$$

$$
\underset {x \in V} {\mathbb {E}} \underset {y \sim M (x)} {\mathbb {E}} d (x, y) \geq \Omega (k).
$$

$$
R (x) = \underset {y \sim M (x)} {\mathbb {E}} d (x, y).
$$

$$
\left| P _ {x} \cap G \right| \geq 2 ^ {2 k} / 1 0. \tag {19}
$$

$$
\Pr \left\{M (y) \in B (y, k / 5 0) \right\} \geq \frac {1}{2}, \tag {20}
$$

$$
1 \geq \Pr \left\{M (x) \in \cup_ {y \in P _ {x} \cap G} B (y, k / 2) \right\} = \sum_ {y \in P _ {x} \cap G} \Pr \left\{M (x) \in B (y, k / 2) \right\} \quad (\text {s i n c e} P _ {x} \text {i s a} k / 2 \text {- p a c k i n g})
$$

$$
\geq \sum_ {y \in P _ {x} \cap G} \exp (- k) \Pr (M (y) \in B (y, k / 2))
$$

$$
= \frac {2 ^ {2 k}}{1 0} \cdot \frac {\exp (- k)}{2} > 1.
$$

$$
\sup  _ {x, y \in V} \max  \left\{\frac {d (x , y)}{d ^ {*} (x , y)}, \frac {d ^ {*} (x , y)}{d (x , y)} \right\} \leq C. \tag {21}
$$
