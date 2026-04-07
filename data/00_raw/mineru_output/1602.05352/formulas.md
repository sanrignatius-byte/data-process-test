$$
R (\hat {Y}) = \frac {1}{U \cdot I} \sum_ {u = 1} ^ {U} \sum_ {i = 1} ^ {I} \delta_ {u, i} (Y, \hat {Y}), \tag {1}
$$

$$
\hat {R} _ {n a i v e} (\hat {Y}) = \frac {1}{| \{(u , i) : O _ {u , i} = 1 \} |} \sum_ {(u, i): O _ {u, i} = 1} \delta_ {u, i} (Y, \hat {Y}). \tag {5}
$$

$$
\mathbb {E} _ {O} \left[ \hat {R} _ {\text {n a i v e}} (\hat {Y}) \right] \neq R (\hat {Y}). \tag {6}
$$

$$
\mathrm {C G}: \quad \delta_ {u, i} (Y, \hat {Y}) = (I / k) \hat {Y} _ {u, i} \cdot Y _ {u, i}. \tag {7}
$$

$$
\mathrm {D C G}: \delta_ {u, i} (Y, \hat {Y}) = \left(I / \log \left(\operatorname {r a n k} \left(\hat {Y} _ {u, i}\right)\right)\right) Y _ {u, i}, \tag {8}
$$

$$
\operatorname {P R E C} @ k: \delta_ {u, i} (Y, \hat {Y}) = (I / k) Y _ {u, i} \cdot \mathbf {1} \left\{\operatorname {r a n k} (\hat {Y} _ {u, i}) \leq k \right\}. \tag {9}
$$

$$
\hat {R} _ {I P S} (\hat {Y} | P) = \frac {1}{U \cdot I} \sum_ {(u, i): O _ {u, i} = 1} \frac {\delta_ {u , i} (Y , \hat {Y})}{P _ {u , i}}. \tag {10}
$$

$$
\begin{array}{l} \mathbb {E} _ {O} \Big [ \hat {R} _ {I P S} (\hat {Y} | P) \Big ] = \frac {1}{U \cdot I} \sum_ {u} \sum_ {i} \mathbb {E} _ {O _ {u, i}} \bigg [ \frac {\delta_ {u , i} (Y , \hat {Y})}{P _ {u , i}} O _ {u, i} \bigg ] \\ = \frac {1}{U \cdot I} \sum_ {u} \sum_ {i} \delta_ {u, i} (Y, \hat {Y}) = R (\hat {Y}). \\ \end{array}
$$

$$
\left| \hat {R} _ {I P S} (\hat {Y} | P) - R (\hat {Y}) \right| \leq \frac {1}{U \cdot I} \sqrt {\frac {\log \frac {2}{\eta}}{2} \sum_ {u , i} \rho_ {u , i} ^ {2}},
$$

$$
\hat {R} _ {S N I P S} (\hat {Y} | P) = \frac {\sum_ {(u , i) : O _ {u , i} = 1} \frac {\delta_ {u , i} (Y , \hat {Y})}{P _ {u , i}}}{\sum_ {(u , i) : O _ {u , i} = 1} \frac {1}{P _ {u , i}}} \tag {11}
$$

$$
\hat {Y} ^ {E R M} = \underset {\hat {Y} \in \mathcal {H}} {\operatorname {a r g m i n}} \left\{\hat {R} _ {I P S} (\hat {Y} | P) \right\}. \tag {12}
$$

$$
\begin{array}{l} R (\hat {Y} ^ {E R M}) \leq \hat {R} _ {I P S} (\hat {Y} ^ {E R M} | P) + \\ \frac {\Delta}{U \cdot I} \sqrt {\frac {\log (2 | \mathcal {H} | / \eta)}{2}} \sqrt {\sum_ {u , i} \frac {1}{P _ {u , i} ^ {2}}} \tag {13} \\ \end{array}
$$

$$
\underset {V, W, A} {\operatorname {a r g m i n}} \left[ \sum_ {O _ {u, i} = 1} \frac {\delta_ {u , i} (Y , V ^ {T} W + A)}{P _ {u , i}} + \lambda \left(\left| \left| V \right| \right| _ {F} ^ {2} + \left| \left| W \right| \right| _ {F} ^ {2}\right) \right] \tag {14}
$$

$$
\operatorname {b i a s} \left(\hat {R} _ {I P S} (\hat {Y} | \hat {P})\right) = \sum_ {u, i} \frac {\delta_ {u , i} (Y , \hat {Y})}{U \cdot I} \left[ 1 - \frac {P _ {u , i}}{\hat {P} _ {u , i}} \right]. \tag {15}
$$

$$
\begin{array}{l} R (\hat {Y} ^ {E R M}) \leq \hat {R} _ {I P S} (\hat {Y} ^ {E R M} | \hat {P}) + \frac {\Delta}{U \cdot I} \sum_ {u, i} \left| 1 - \frac {P _ {u , i}}{\hat {P} _ {u , i}} \right| \\ + \frac {\Delta}{U \cdot I} \sqrt {\frac {\log (2 | \mathcal {H} | / \eta)}{2}} \sqrt {\sum_ {u , i} \frac {1}{\hat {P} _ {u , i} ^ {2}}} \tag {16} \\ \end{array}
$$

$$
P _ {u, i} = P \left(O _ {u, i} = 1 \mid X, X ^ {h i d}, Y\right) \tag {17}
$$

$$
P \left(O _ {u, i} = 1 \mid Y _ {u, i} = r\right) = \frac {P (Y = r \mid O = 1) P (O = 1)}{P (Y = r)}. \tag {18}
$$
