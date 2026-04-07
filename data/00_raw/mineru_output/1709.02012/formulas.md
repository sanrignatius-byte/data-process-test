$$
c _ {f n} \left(h _ {t}\right) = \left(1 - \mu_ {t}\right) / \mu_ {t} c _ {f p} \left(h _ {t}\right). \tag {1}
$$

$$
g _ {t} \left(h _ {t}\right) = a _ {t} c _ {f p} \left(h _ {t}\right) + b _ {t} c _ {f n} \left(h _ {t}\right) \tag {2}
$$

$$
\tilde {h} _ {2} (\mathbf {x}) = \left\{ \begin{array}{l l} h ^ {\mu_ {2}} (\mathbf {x}) = \mu_ {2} & \text {w i t h p r o b a b i l i t y} \alpha \\ h _ {2} (\mathbf {x}) & \text {w i t h p r o b a b i l i t y} 1 - \alpha \end{array} \right. \tag {3}
$$

$$
\epsilon \left(h _ {t}\right) = \int_ {0} ^ {1} \left| \underset {(\mathbf {x}, y) \sim G _ {t}} {\mathrm {P}} [ y = 1 \mid h (\mathbf {x}) = p ] - p \right| \underset {(\mathbf {x}, y) \sim G _ {t}} {\mathrm {P}} [ h (\mathbf {x}) = p ] d p. \tag {S1}
$$

$$
\left| \mu_ {t} c _ {f n} (h _ {t}) - (1 - \mu_ {t}) c _ {f p} (h _ {t}) \right| \leq 2 \delta_ {c a l}.
$$

$$
\begin{array}{l} c _ {f p} (h _ {t}) = \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \mid y = 0 \right] \\ = \int_ {0} ^ {1} p \underset {G _ {t}} {\mathrm {P}} \left[ h _ {t} (\mathbf {x}) = p \mid y = 0 \right] d p \\ = \int_ {0} ^ {1} p \frac {1 - \mathrm {P} _ {G _ {t}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]}{1 - \mathrm {P} _ {G _ {t}} [ y = 1 ]} \mathrm {P} _ {G _ {t}} [ h _ {t} (\mathbf {x}) = p ] d p \\ = \frac {1}{1 - \mu_ {t}} \int_ {0} ^ {1} p \left(1 - \underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]\right) \underset {G _ {t}} {\mathrm {P}} [ h _ {t} (\mathbf {x}) = p ] d p \tag {S2} \\ \end{array}
$$

$$
\begin{array}{l} \int_ {0} ^ {1} p \cdot \underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ] \cdot \underset {G _ {t}} {\mathrm {P}} [ h _ {t} (\mathbf {x}) = p ] d p \\ = \int_ {0} ^ {1} p (p + \underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ] - p) \underset {G _ {t}} {\mathrm {P}} [ h _ {t} (\mathbf {x}) = p ] d p \\ \leq \int_ {0} ^ {1} \left(p ^ {2} + | \mathrm {P} _ {G _ {t}} [ y = 1 | h _ {t} (\mathbf {x}) = p ] - p |\right) \mathrm {P} _ {G _ {t}} [ h _ {t} (\mathbf {x}) = p ] d p \\ \leq \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] + \delta_ {c a l} \\ \end{array}
$$

$$
\begin{array}{l} \int_ {0} ^ {1} p \cdot \underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ] \underset {G _ {t}} {\mathrm {P}} [ h _ {t} (\mathbf {x}) = p ] d p \\ \geq \int_ {0} ^ {1} \left(p ^ {2} - | P _ {G _ {t}} [ y = 1 | h _ {t} (\mathbf {x}) = p ] - p |\right) P _ {G _ {t}} [ h _ {t} (\mathbf {x}) = p ] d p \\ \geq \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] - \delta_ {c a l} \\ \end{array}
$$

$$
\begin{array}{l} \frac {1}{1 - \mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] - \delta_ {c a l}\right) \leq c _ {f p} \left(h _ {t}\right) \\ \leq \frac {1}{1 - \mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] + \delta_ {c a l}\right) \tag {S3} \\ \end{array}
$$

$$
\begin{array}{l} c _ {f n} \left(h _ {t}\right) = \underset {G _ {t}} {\mathbb {E}} \left[ 1 - h _ {t} (\mathbf {x}) \mid y = 0 \right] \\ = \int_ {0} ^ {1} (1 - p) \mathrm {P} _ {G _ {t}} \left[ h _ {t} (\mathbf {x}) = p \mid y = 0 \right] d p \\ = \int_ {0} ^ {1} (1 - p) \frac {\mathrm {P} _ {G _ {t}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]}{\mathrm {P} _ {G _ {t}} [ y = 1 ]} \mathrm {P} _ {G _ {t}} [ h _ {t} (\mathbf {x}) = p ] d p. \\ = \frac {1}{\mu_ {t}} \int_ {0} ^ {1} (1 - p) \left(\underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]\right) \underset {G _ {t}} {\mathrm {P}} [ h _ {t} (\mathbf {x}) = p ] d p. \\ \end{array}
$$

$$
\begin{array}{l} (1 - p) \left(\underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]\right) \\ = (1 - p) \left(p + \underset {G _ {t}} {\mathrm {P}} \left[ y = 1 \mid h _ {t} (\mathbf {x}) = p \right] - p\right) \\ \leq p (1 - p) + \left| \begin{array}{c} \mathrm {P} \\ G _ {t} \end{array} \right. [ y = 1 \mid h _ {t} (\mathbf {x}) = p ] - p | \\ \end{array}
$$

$$
\begin{array}{l} (1 - p) \left(\underset {G _ {t}} {\mathrm {P}} [ y = 1 \mid h _ {t} (\mathbf {x}) = p ]\right) \\ \geq p (1 - p) - \left| \begin{array}{c} \mathrm {P} \\ G _ {t} \end{array} \right. [ y = 1 \mid h _ {t} (\mathbf {x}) = p ] - p | \\ \end{array}
$$

$$
\begin{array}{l} \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] - \delta_ {c a l}\right) \leq c _ {f n} \left(h _ {t}\right) \\ \leq \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] + \delta_ {c a l}\right) \tag {S4} \\ \end{array}
$$

$$
\begin{array}{l} c _ {f n} \left(h _ {t}\right) \leq \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] + \delta_ {c a l}\right) \\ = \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] - \delta_ {c a l} + 2 \delta_ {c a l}\right) \\ \leq \frac {1}{\mu_ {t}} \left(\left(1 - \mu_ {t}\right) c _ {f p} \left(h _ {t}\right) + 2 \delta_ {c a l}\right) \\ = \frac {1 - \mu_ {t}}{\mu_ {t}} c _ {f p} (h _ {t}) + \frac {2 \delta_ {c a l}}{\mu_ {t}} \\ \end{array}
$$

$$
\begin{array}{l} c _ {f n} \left(h _ {t}\right) \geq \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right] - \delta_ {c a l}\right) \\ \geq \frac {1}{\mu_ {t}} \left(\left(1 - \mu_ {t}\right) c _ {f p} \left(h _ {t}\right) - 2 \delta_ {c a l}\right) \\ = \frac {1 - \mu_ {t}}{\mu_ {t}} c _ {f p} (h _ {t}) - \frac {2 \delta_ {c a l}}{\mu_ {t}} \\ \end{array}
$$

$$
c _ {f p} \left(h _ {t} ^ {*}\right) = \frac {1}{1 - \mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right]\right) \tag {S5}
$$

$$
c _ {f n} \left(h _ {t} ^ {*}\right) = \frac {1}{\mu_ {t}} \left(\underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) \right] - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (\mathbf {x}) ^ {2} \right]\right) \tag {S6}
$$

$$
c _ {f n} \left(h _ {t} ^ {*}\right) = \frac {1 - \mu_ {t}}{\mu_ {t}} c _ {f p} \left(h _ {t}\right). \tag {S7}
$$

$$
g _ {t} \left(h _ {t}\right) = a _ {t} c _ {f p} \left(h _ {t}\right) + b _ {t} c _ {f n} \left(h _ {t}\right)
$$

$$
g _ {t} (h) = a _ {t} c _ {f p} (h _ {t}) + b _ {t} c _ {f n} (h _ {t}).
$$

$$
\begin{array}{l} g _ {t} (h _ {t}) = a _ {t} c _ {f p} (h _ {t}) + b _ {t} c _ {f n} (h _ {t}) \\ = \left(\frac {a _ {t}}{1 - \mu_ {t}} + \frac {b _ {t}}{\mu_ {t}}\right) \left(\underset {G _ {t}} {\mathbb {E}} [ h _ {t} (x) ] - \underset {G _ {t}} {\mathbb {E}} [ h _ {t} (x) ^ {2} ]\right) \\ = \left(\frac {a _ {t}}{1 - \mu_ {t}} + \frac {b _ {t}}{\mu_ {t}}\right) \left(\mu_ {t} - \underset {G _ {t}} {\mathbb {E}} \left[ h _ {t} (x) ^ {2} \right]\right). \\ \end{array}
$$

$$
\begin{array}{l} h _ {t} ^ {\max } = \underset {h \in \mathcal {H} _ {t} ^ {*}} {\operatorname {a r g m a x}} \left[ \left(\frac {a _ {t}}{1 - \mu_ {t}} + \frac {b _ {t}}{\mu_ {t}}\right) \left(\mu_ {t} - \underset {G _ {t}} {\mathbb {E}} [ h (x) ^ {2} ]\right) \right] \\ = \operatorname * {a r g m a x} _ {h \in \mathcal {H} _ {t} ^ {*}} \left[ - \underset {G _ {t}} {\mathbb {E}} \left[ h (x) ^ {2} \right] \right] \\ = \operatorname * {a r g m i n} _ {h \in \mathcal {H} _ {t} ^ {*}} \left[ \underset {G _ {t}} {\mathbb {E}} \left[ h (x) ^ {2} \right] \right] \\ = \operatorname * {a r g m i n} _ {h \in \mathcal {H} _ {t} ^ {*}} \left[ \underset {G _ {t}} {\mathbb {E}} \left[ h (x) ^ {2} \right] - \mu_ {t} ^ {2} \right] \\ \end{array}
$$

$$
h _ {t} ^ {\max } (\mathbf {x}) = h ^ {\mu_ {t}} (\mathbf {x}) = \mu_ {t}.
$$

$$
g _ {2} (\tilde {h} _ {2}) = (1 - \alpha) g _ {2} (h _ {2}) + \alpha g _ {2} \left(h ^ {\mu_ {2}}\right)
$$

$$
\begin{array}{l} g _ {2} (\tilde {h} _ {2}) = a _ {2} c _ {f p} (\tilde {h} _ {2}) + b _ {2} c _ {f n} (\tilde {h} _ {2}) \\ = a _ {2} \underset {G _ {2}} {\mathbb {E}} \left[ 1 - \tilde {h} _ {2} (\mathbf {x}) \mid y = 1 \right] + b _ {2} \underset {G _ {2}} {\mathbb {E}} \left[ \tilde {h} _ {2} (\mathbf {x}) \mid y = 0 \right] \\ = a _ {2} \underset {B, G _ {2}} {\mathbb {E}} \left[ 1 - \left[ (1 - B) h _ {2} (\mathbf {x}) + B h ^ {\mu_ {2}} (\mathbf {x}) \right] \mid y = 1 \right] + b _ {2} \underset {B, G _ {2}} {\mathbb {E}} \left[ \left[ (1 - B) h _ {2} (\mathbf {x}) + B h ^ {\mu_ {2}} (\mathbf {x}) \right] \mid y = 0 \right] \\ = a _ {2} \underset {B, G _ {2}} {\mathbb {E}} [ (1 - B) (1 - h _ {2} (\mathbf {x})) \mid y = 1 ] + a _ {2} \underset {B, G _ {2}} {\mathbb {E}} [ B (1 - h ^ {\mu_ {2}} (\mathbf {x})) \mid y = 1 ] \\ + b _ {2} \underset {B, G _ {2}} {\mathbb {E}} [ (1 - B) h _ {2} (\mathbf {x}) \mid y = 0 ] + b _ {2} \underset {B, G _ {2}} {\mathbb {E}} \left[ B h ^ {\mu_ {2}} (\mathbf {x}) \mid y = 0 \right] \\ = a _ {2} \underset {B} {\mathbb {E}} [ 1 - B ] \underset {G _ {2}} {\mathbb {E}} [ 1 - h _ {2} (\mathbf {x}) \mid y = 1 ] + a _ {2} \underset {B} {\mathbb {E}} [ B ] \underset {G _ {2}} {\mathbb {E}} [ 1 - h ^ {\mu_ {2}} (\mathbf {x}) \mid y = 1 ] \\ + b _ {2} \underset {B} {\mathbb {E}} [ 1 - B ] \underset {G _ {2}} {\mathbb {E}} [ h _ {2} (\mathbf {x}) \mid y = 0 ] + b _ {2} \underset {B} {\mathbb {E}} [ B ] \underset {G _ {2}} {\mathbb {E}} [ h ^ {\mu_ {2}} (\mathbf {x}) \mid y = 0 ] \\ = a _ {2} (1 - \alpha) c _ {f p} (h _ {2}) + b _ {2} (1 - \alpha) c _ {f n} (h _ {2}) + a _ {2} (\alpha) c _ {f p} \left(h ^ {\mu_ {2}}\right) + b _ {2} (\alpha) c _ {f n} \left(h ^ {\mu_ {2}}\right) \\ = (1 - \alpha) g _ {2} \left(h _ {2}\right) + \alpha g _ {2} \left(h ^ {\mu_ {2}}\right). \\ \end{array}
$$

$$
\begin{array}{l} c _ {f p} \left(h _ {t} ^ {\prime}\right) \geq \frac {\mu_ {t}}{1 - \mu_ {t}} c _ {f n} \left(h _ {t} ^ {\prime}\right) - \frac {2 \delta_ {c a l}}{1 - \mu_ {t}} \\ \geq \frac {\mu_ {t}}{1 - \mu_ {t}} c _ {f n} (h _ {t}) - \frac {2 \delta_ {c a l}}{1 - \mu_ {t}} \\ \geq c _ {f p} (h _ {t}) - \frac {4 \delta_ {c a l}}{1 - \mu_ {t}} \\ \end{array}
$$

$$
\begin{array}{l} c _ {f n} \left(h _ {t} ^ {\prime}\right) \geq \frac {1 - \mu_ {t}}{\mu_ {t}} c _ {f p} \left(h _ {t} ^ {\prime}\right) - \frac {2 \delta_ {c a l}}{\mu_ {t}} \\ \geq \frac {1 - \mu_ {t}}{\mu_ {t}} c _ {f p} (h _ {t}) - \frac {2 \delta_ {c a l}}{\mu_ {t}} \\ \geq c _ {f n} \left(h _ {t}\right) - \frac {4 \delta_ {c a l}}{\mu_ {t}} \\ \end{array}
$$

$$
\begin{array}{l} \epsilon (\tilde {h} _ {2}) = \underset {B, G _ {2}} {\mathbb {E}} \left| \underset {G _ {2}} {\mathrm {P}} [ y = 1 \mid \tilde {h} _ {2} (\mathbf {x}) = p ] - p \right| \\ = \int_ {0} ^ {1} \left| _ {G _ {2}} \left[ y = 1 \mid \tilde {h} _ {2} (\mathbf {x}) = p \right] - p \right| _ {G _ {2}} \left[ \tilde {h} _ {2} (\mathbf {x}) = p \right] d p \\ \end{array}
$$

$$
\begin{array}{l} \underset {G _ {2}} {\mathrm {P}} [ y = 1 \mid \tilde {h} _ {2} (\mathbf {x}) = p ] = (1 - \beta) \underset {G _ {2}} {\mathrm {P}} [ y = 1 \mid h _ {2} (\mathbf {x}) = p ] + \beta \underset {G _ {2}} {\mathrm {P}} [ y = 1 \mid h ^ {\mu_ {2}} (\mathbf {x}) = p ] \\ = (1 - \beta) \underset {G _ {2}} {\mathrm {P}} [ y = 1 \mid h _ {2} (\mathbf {x}) = p ] + \beta p \\ \end{array}
$$

$$
\begin{array}{l} \epsilon (\tilde {h} _ {2}) = \int_ {0} ^ {1} \left| \mathrm {P} _ {G _ {2}} [ y = 1 | \tilde {h} _ {2} (\mathbf {x}) = p ] - p \right| \mathrm {P} _ {G _ {2}} [ \tilde {h} _ {2} (\mathbf {x}) = p ] d p \\ \leq \int_ {0} ^ {1} \left| \mathrm {P} _ {G _ {2}} [ y = 1 \mid h _ {2} (\mathbf {x}) = p ] - p \right| \mathrm {P} _ {G _ {2}} [ h _ {2} (\mathbf {x}) = p ] d p \\ = \epsilon (h _ {2}) \\ \end{array}
$$

$$
A = \left[ \begin{array}{c c c c} 1 & - \frac {\mu_ {1}}{1 - \mu_ {1}} & 0 & 0 \\ 0 & 0 & 1 & - \frac {\mu_ {2}}{1 - \mu_ {2}} \\ a _ {1} & b _ {1} & - a _ {2} & - b _ {2} \\ a _ {1} ^ {\prime} & b _ {1} ^ {\prime} & - a _ {2} ^ {\prime} & - b _ {2} ^ {\prime} \end{array} \right].
$$

$$
\vec {q} = \left[ c _ {f p} (h _ {1}) c _ {f n} (h _ {1}) c _ {f p} (h _ {2}) c _ {f n} (h _ {2}) \right] ^ {\top}.
$$

$$
c _ {f p} (h _ {t}) \leq L \cdot \max  \left\{\frac {2 \delta_ {c a l}}{1 - \mu_ {1}}, \frac {2 \delta_ {c a l}}{1 - \mu_ {2}}, \delta_ {c o s t} \right\}
$$

$$
c _ {f n} \left(h _ {t}\right) \leq L \cdot \max  \left\{\frac {2 \delta_ {c a l}}{1 - \mu_ {1}}, \frac {2 \delta_ {c a l}}{1 - \mu_ {2}}, \delta_ {c o s t} \right\}
$$

$$
\left| \mu_ {t} c _ {f n} (h _ {t}) - \left(1 - \mu_ {t}\right) c _ {f p} (h _ {t}) \right| \leq 2 \delta_ {c a l}.
$$

$$
| A \vec {q} | \leq \left[ \begin{array}{c} \frac {2 \delta_ {c a l}}{1 - \mu_ {1}} \\ \frac {2 \delta_ {c a l}}{1 - \mu_ {2}} \\ \delta_ {c o s t} \\ \delta_ {c o s t} \end{array} \right],
$$

$$
\widehat {A} \vec {q} \leq \vec {\nu}
$$

$$
\vec {q} \leq \widehat {A} ^ {- 1} \vec {\nu}.
$$

$$
\| \vec {q} \| _ {\infty} \leq \| \widehat {A} ^ {- 1} \vec {v} \| _ {\infty} \leq \| \widehat {A} ^ {- 1} \| _ {\infty} \| \vec {v} \| _ {\infty}.
$$

$$
\| \widehat {A} ^ {- 1} \| _ {\infty} \leq \max  _ {j} \sum_ {i = 1} ^ {4} | d _ {i j} | \leq 1 6 M ^ {3} D ^ {4} = L
$$

$$
\| \tilde {q} \| _ {\infty} \leq L \| \nu \| _ {\infty}
$$

$$
h _ {t} ^ {e o} (\mathbf {x}) = \left\{ \begin{array}{l l} (1 - h _ {t} (\mathbf {x})) B _ {\mathrm {p 2 n}} ^ {(t)} + h _ {t} (\mathbf {x}) \left(1 - B _ {\mathrm {p 2 n}} ^ {(t)}\right) & h _ {t} (\mathbf {x}) \geq 0. 5 \\ (1 - h _ {t} (\mathbf {x})) B _ {\mathrm {n 2 p}} ^ {(t)} + h _ {t} (\mathbf {x}) \left(1 - B _ {\mathrm {n 2 p}} ^ {(t)}\right) & h _ {t} (\mathbf {x}) <   0. 5 \end{array} \right.
$$

$$
\min  _ { \begin{array}{c} q ^ {(1)}, q _ {\mathrm {p 2 n}}, q _ {\mathrm {n 2 p}}, q _ {\mathrm {p 2 n}} ^ {(2)} \end{array} } \mathcal {L} (h _ {1} ^ {e o}) + \mathcal {L} (h _ {2} ^ {e o})
$$

$$
c _ {f p} (h _ {2} ^ {e o}) = c _ {f n} (h _ {2} ^ {e o})
$$

$$
\mathcal {L} \left(h _ {t}\right) = \underset {G _ {t}} {\mathrm {P}} \left[ h _ {t} (\mathbf {x}) \geq 0. 5 \mid y = 0 \right] + \underset {G _ {t}} {\mathrm {P}} \left[ h _ {t} (\mathbf {x}) <   0. 5 \mid y = 1 \right].
$$
