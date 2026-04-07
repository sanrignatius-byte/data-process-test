$$
L (q *) = A; C = C \backslash \{q * \};
$$

$$
i f A i s \text {y e s} ” t h e n
$$

$$
\begin{array}{l} C h l d r = \{q \in C h i l d r e n (q *, \mathcal {G}) \colon L (q) = \emptyset \}; \\ C = C \cup C h l d r; \end{array}
$$

$$
\begin{array}{l} D e s = \{q \in D e s c e n d a n t s (q *, \mathcal {G}) \colon L (q) = \emptyset \}; \\ L (q) = \text {" n o " \forall q \in D e s ;} \\ C = C \backslash D e s; \end{array}
$$

$$
\operatorname {e r r o r} = \frac {1}{N} \sum_ {i = 1} ^ {N} \min  _ {j} d _ {i j} \tag {1}
$$

$$
d _ {i j} = \max  \left(d \left(c _ {i j}, C _ {i}\right), \min  _ {k} d \left(b _ {i j}, B _ {i k}\right)\right) \tag {2}
$$

$$
\operatorname {R e c a l l} (t) = \frac {\sum_ {i j} 1 [ s _ {i j} \geq t ] z _ {i j}}{N} \tag {3}
$$

$$
P r e c i s i o n (t) = \frac {\sum_ {i j} 1 \left[ s _ {i j} \geq t \right] z _ {i j}}{\sum_ {i j} 1 \left[ s _ {i j} \geq t \right]} \tag {4}
$$

$$
\operatorname {t h r} (B) = \min  \left(0. 5, \frac {w h}{(w + 1 0) (h + 1 0)}\right) \tag {5}
$$

$$
\mathrm {C P L} = \frac {\sum_ {i} \sum_ {j \neq i} I O U \left(B _ {i} , B _ {j}\right) \geq 0 . 5}{N (N - 1)} \tag {6}
$$

$$
\mathrm {C L U T T E R} = \log_ {2} \left(\frac {1}{M} \sum_ {m} \mathrm {O B J} (m)\right) \tag {7}
$$
