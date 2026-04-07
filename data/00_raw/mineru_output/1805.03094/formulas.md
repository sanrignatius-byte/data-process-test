$$
\sum_ {i = 1} ^ {N} \left(y _ {i} - \bar {y}\right) ^ {2} = \sum_ {b \in P _ {X _ {c}}} N _ {b} \left(\bar {y} _ {b} - \bar {y}\right) ^ {2} + \sum_ {b \in P _ {X _ {c}}} \sum_ {i = 1} ^ {N _ {b}} \left(y _ {b, i} - \bar {y} _ {b}\right) ^ {2}, \tag {1}
$$

$$
R ^ {2} = \frac {\sum_ {b \in P _ {X _ {c}}} N _ {b} (\bar {y} _ {b} - \bar {y}) ^ {2}}{S S T} \tag {2}
$$

$$
R ^ {2} (s; X _ {c}) = \frac {N _ {b _ {1}} (\bar {y} _ {b _ {1}} - \bar {y}) ^ {2} + N _ {b _ {2}} (\bar {y} _ {b _ {2}} - \bar {y}) ^ {2}}{S S T}, \tag {3}
$$

$$
\begin{array}{l} \Delta R ^ {2} (s | P _ {X _ {c}}; X _ {c}) = \frac {1}{S S T} \left(N _ {b _ {i _ {1}}} (\bar {y} _ {b _ {i _ {1}}}) ^ {2} + N _ {b _ {i _ {2}}} (\bar {y} _ {b _ {i _ {2}}}) ^ {2} \right. \\ \left. - N _ {b _ {i}} \left(\bar {y} _ {b _ {i}}\right) ^ {2}\right) \tag {4} \\ \end{array}
$$

$$
\mathbb {E} [ Y | X _ {j} = x _ {j} ] = f (\alpha + \beta X _ {j}), \tag {5}
$$

$$
\mathbb {E} \left[ Y \mid X _ {j} = x _ {j}, X _ {c} = x _ {c} \right] = f \left(\alpha \left(x _ {c}\right) + \beta \left(x _ {c}\right) X _ {j}\right). \tag {6}
$$

$$
\mathbb {E} [ Y | X _ {j} = x _ {j} ] = f (\alpha + \beta X _ {j}) = \frac {1}{1 + e ^ {- (\alpha + \beta X _ {j})}} \tag {7}
$$

$$
\mathcal {L} (\mathcal {M} | x) = \prod_ {i = 1} ^ {N} y _ {i} \times \left(P _ {\mathcal {M}} \left(x _ {i}\right)\right) + \left(1 - y _ {i}\right) \times \left(1 - P _ {\mathcal {M}} \left(x _ {i}\right)\right) \tag {8}
$$

$$
\log \mathcal {L} (\mathcal {M} | x) = \sum_ {i = 1} ^ {N} y _ {i} \times \log \left(P _ {\mathcal {M}} \left(x _ {i}\right)\right) + \left(1 - y _ {i}\right) \times \log \left(1 - P _ {\mathcal {M}} \left(x _ {i}\right)\right) \tag {9}
$$

$$
D \left(\mathcal {M} _ {1}, \mathcal {M} _ {0}\right) = 2 \times \left[ \log \mathcal {L} \left(\mathcal {M} _ {1} | x\right) - \log \mathcal {L} \left(\mathcal {M} _ {0} | x\right) \right] \tag {10}
$$

$$
D \left(\mathcal {M} _ {1}, \mathcal {M} _ {0}\right) = 2 \times \sum_ {i = 1} ^ {N} y _ {i} \times \log \left(\frac {\hat {y} _ {i}}{\bar {y}}\right) + \left(1 - y _ {i}\right) \times \log \left(\frac {1 - \hat {y} _ {i}}{1 - \bar {y}}\right) \tag {11}
$$

$$
2 \times \sum_ {b \in P _ {X _ {c}}} \sum_ {i = 1} ^ {N _ {b}} y _ {b, i} \times \log \left(\frac {\hat {y} _ {b , i}}{\bar {y} _ {b}}\right) + (1 - y _ {b, i}) \times \log \left(\frac {1 - \hat {y} _ {b , i}}{1 - \bar {y} _ {b}}\right), \tag {12}
$$

$$
R _ {M c F a d d e n} ^ {2} = 1 - \frac {\log \mathcal {L} _ {f u l l}}{\log \mathcal {L} _ {n u l l}} \tag {13}
$$

$$
1 - \frac {\sum_ {b \in P _ {X _ {c}}} \sum_ {i = 1} ^ {N _ {b}} y _ {b , i} \times \log \left(y _ {\hat {b} , i}\right) + \left(1 - y _ {b , i}\right) \times \log \left(1 - y _ {\hat {b} , i} ^ {\prime}\right)}{\sum_ {i = 1} ^ {N} y _ {i} \times \log (\bar {y}) + \left(1 - y _ {i}\right) \times \log (1 - \bar {y})} \tag {14}
$$
