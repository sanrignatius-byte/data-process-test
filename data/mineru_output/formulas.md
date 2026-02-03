$$
\mathcal {E} \leftarrow \pi_ {\mathrm {e}} (\mathcal {C} \oplus s \oplus \{t \}), \tag {1}
$$

$$
w \left(s _ {i}\right) = \mathbf {1} \left(s _ {i}\right) \cdot \delta \left(s _ {i}\right), \tag {2}
$$

$$
\delta \left(s _ {i}\right) = \frac {\left| \mathcal {T} _ {i} \right|}{\sum_ {j = 1} ^ {n} \left| \mathcal {T} _ {j} \right|}. \tag {3}
$$

$$
\mathcal {L} _ {\mathrm {a l p o}} \left(\pi_ {\theta}; \pi_ {\mathrm {r e f}}\right) = - \mathbb {E} _ {\left(x, S (x)\right) \sim \mathbb {D} _ {\mathrm {a l p o}}} \left[ \sum_ {i = 1} ^ {n} w \left(s _ {i}\right) \cdot \log \sigma \left(\beta_ {i} \log \frac {\pi_ {\theta} \left(t _ {i} ^ {\left(\mathrm {c}\right)} \mid p _ {i}\right)}{\pi_ {\mathrm {r e f}} \left(t _ {i} ^ {\left(\mathrm {c}\right)} \mid p _ {i}\right)} - \beta_ {i} \log \frac {\pi_ {\theta} \left(t _ {i} ^ {\left(\mathrm {r}\right)} \mid p _ {i}\right)}{\pi_ {\mathrm {r e f}} \left(t _ {i} ^ {\left(\mathrm {r}\right)} \mid p _ {i}\right)}\right) \right]. \tag {4}
$$

$$
\beta_ {i} = \frac {r \left(s _ {i} , t _ {i} ^ {\left(\mathrm {c}\right)}\right) - r \left(s _ {i} , t _ {i} ^ {\left(\mathrm {r}\right)}\right)}{\max  \left\{r \left(s _ {j} , t _ {j} ^ {\left(\mathrm {c}\right)}\right) - r \left(s _ {j} , t _ {j} ^ {\left(\mathrm {r}\right)}\right) \mid j \in [ n ] \right\}}, \tag {5}
$$

$$
p _ {i} = x, \hat {t} _ {1}, \dots , \hat {t} _ {i - 1}, \quad \hat {t} _ {j} \leftarrow \operatorname {M i x} \left(t _ {j} ^ {(c)}, t _ {j} \sim \mathcal {T}, \lambda\right). \tag {6}
$$

$$
\mathcal {L} _ {\mathrm {a l p o}} \left(\pi_ {\theta}; \pi_ {\text {r e f}}\right) = - \mathbb {E} _ {\left(x, S (x)\right) \sim \mathbb {D} _ {\mathrm {a l p o}}} \left(\sum_ {i = 1} ^ {n} w \left(s _ {i}\right) \cdot \mathcal {L} _ {\mathrm {p o}} \left(s _ {i}\right)\right), \tag {7}
$$

$$
\mathcal {L} _ {\mathrm {p o}} (s _ {i}) =
$$

$$
\frac {1}{| \mathcal {T} _ {i} |} \sum_ {j} \left[ \min  \left(\frac {\pi_ {\theta} \left(t _ {i} ^ {j} \mid p _ {i}\right)}{\pi_ {\operatorname {r e f}} \left(t _ {i} ^ {j} \mid p _ {i}\right)} A _ {j}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} \left(t _ {i} ^ {j} \mid p _ {i}\right)}{\pi_ {\operatorname {r e f}} \left(t _ {i} ^ {j} \mid p _ {i}\right)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {j}\right) - \eta \mathrm {K L} \left(\pi_ {\theta} \mid \mid \pi_ {\text {r e f}}\right) \right], \tag {8}
$$

$$
A _ {j} = \frac {\mathcal {E} _ {i} ^ {j} - \operatorname {m e a n} \left(\mathcal {E} _ {i}\right)}{\operatorname {s t d} \left(\mathcal {E} _ {i}\right)}, \quad \mathrm {K L} \left(\pi_ {\theta} \| \pi_ {\text {r e f}}\right) = \frac {\pi_ {\text {r e f}} \left(t _ {i} ^ {j} \mid p _ {i}\right)}{\pi_ {\theta} \left(t _ {i} ^ {j} \mid p _ {i}\right)} - \log \frac {\pi_ {\text {r e f}} \left(t _ {i} ^ {j} \mid p _ {i}\right)}{\pi_ {\theta} \left(t _ {i} ^ {j} \mid p _ {i}\right)} - 1, \tag {9}
$$

$$
\begin{array}{l} \delta_ {i} = - \mathcal {E} _ {i} + \gamma V _ {\phi} (p _ {i + 1}) - V _ {\phi} (p _ {i}), \\ A _ {i} = \sum_ {l = 0} ^ {n - i - 1} (\gamma \lambda) ^ {l} \delta_ {i + l}. \end{array} \tag {10}
$$

$$
\mathcal {L} _ {\mathrm {c l i p}} (\theta) = - \mathbb {E} _ {x \sim \mathbb {D} _ {\mathrm {a l p o}}} \left[ \sum_ {i = 1} ^ {n} \min  \left(\frac {\pi_ {\theta} \left(t _ {i} \mid p _ {i}\right)}{\pi_ {\mathrm {o l d}} \left(t _ {i} \mid p _ {i}\right)} A _ {i}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} \left(t _ {i} \mid p _ {i}\right)}{\pi_ {\mathrm {o l d}} \left(t _ {i} \mid p _ {i}\right)}, 1 - \epsilon , 1 + \epsilon\right) A _ {i}\right) \right]. \tag {11}
$$

$$
\mathcal {L} _ {V} (\phi) = \mathbb {E} _ {x \sim \mathbb {D} _ {\mathrm {a l p o}}} \left[ \sum_ {i = 1} ^ {n} \left(V _ {\phi} \left(p _ {i}\right) - \hat {V} _ {i}\right) ^ {2} \right]. \tag {12}
$$

$$
\mathcal {L} _ {\text {a l p o}} \left(\pi_ {\theta}; \pi_ {\text {r e f}}\right) = - \mathbb {E} _ {\left(x, y _ {1: n} ^ {w}, y _ {1: n} ^ {l}\right) \sim \mathbb {D}} \left[ \sum_ {i = 1} ^ {n} \mathbf {1} \left(p _ {i}\right) \cdot \log \sigma \left(\beta \log \frac {\pi_ {\theta} \left(y _ {i} ^ {w} \mid p _ {i}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {w} \mid p _ {i}\right)} - \beta \log \frac {\pi_ {\theta} \left(y _ {i} ^ {l} \mid p _ {i}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {l} \mid p _ {i}\right)}\right) \right], \tag {13}
$$

$$
\pi_ {\theta} (y \mid x) = \prod_ {i = 1} ^ {n} \pi_ {\theta} \left(y _ {i} \mid x, y _ {1}, \dots , y _ {i - 1}\right). \tag {14}
$$

$$
\theta^ {*} = \arg \max  _ {\theta} \mathbb {E} _ {x \sim p (x)} \left[ \mathbb {E} _ {y \sim \pi_ {\theta} (\cdot | x)} \left[ \sum_ {i = 1} ^ {n} r \left(x _ {i}, y _ {i}\right) \right] \right], \tag {15}
$$

$$
\pi_ {\theta} \left(y ^ {w} \mid x\right) = \prod_ {i = 1} ^ {n} \pi_ {\theta} \left(y _ {i} ^ {w} \mid x, y _ {1} ^ {w}, \dots , y _ {i - 1} ^ {w}\right), \pi_ {\theta} \left(y ^ {l} \mid x\right) = \prod_ {i = 1} ^ {n} \pi_ {\theta} \left(y _ {i} ^ {l} \mid x, y _ {1} ^ {l}, \dots , y _ {i - 1} ^ {l}\right). \tag {16}
$$

$$
\mathcal {L} _ {\mathrm {D P O}} \left(\pi_ {\theta}; \pi_ {\text {r e f}}\right) = - \mathbb {E} _ {\left(x, y ^ {w}, y ^ {l}\right) \sim \mathbb {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} \left(y ^ {w} \mid x\right)}{\pi_ {\text {r e f}} \left(y ^ {w} \mid x\right)} - \beta \log \frac {\pi_ {\theta} \left(y ^ {l} \mid x\right)}{\pi_ {\text {r e f}} \left(y ^ {l} \mid x\right)}\right) \right], \tag {17}
$$

$$
\log \frac {\pi_ {\theta} \left(y ^ {w} \mid x\right)}{\pi_ {\text {r e f}} \left(y ^ {w} \mid x\right)} - \log \frac {\pi_ {\theta} \left(y ^ {l} \mid x\right)}{\pi_ {\text {r e f}} \left(y ^ {l} \mid x\right)} = \sum_ {i = 1} ^ {n} \left[ \log \frac {\pi_ {\theta} \left(y _ {i} ^ {w} \mid x , y _ {1 : i - 1} ^ {w}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {w} \mid x , y _ {1 : i - 1} ^ {w}\right)} - \log \frac {\pi_ {\theta} \left(y _ {i} ^ {l} \mid x , y _ {1 : i - 1} ^ {l}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {l} \mid x , y _ {1 : i - 1} ^ {l}\right)} \right], \tag {18}
$$

$$
\mathcal {L} _ {\text {a l p o}} \left(\pi_ {\theta}; \pi_ {\text {r e f}}\right) = - \mathbb {E} _ {\left(x, y _ {1: n} ^ {w}, y _ {1: n} ^ {l}\right) \sim \mathbb {D}} \left[ \sum_ {i = 1} ^ {n} w \left(p _ {i}\right) \cdot \log \sigma \left(\beta_ {i} \log \frac {\pi_ {\theta} \left(y _ {i} ^ {w} \mid p _ {i}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {w} \mid p _ {i}\right)} - \beta_ {i} \log \frac {\pi_ {\theta} \left(y _ {i} ^ {l} \mid p _ {i}\right)}{\pi_ {\text {r e f}} \left(y _ {i} ^ {l} \mid p _ {i}\right)}\right) \right], \tag {19}
$$

$$
\mathbb {E} _ {p \sim d _ {\theta *}} \left[ \mathcal {L} _ {\text {a l p o}} \left(\pi_ {\theta} (\cdot \mid p)\right) \right], \tag {20}
$$
