$$
e r r o r (Y, Z) = \| Y - Z \| _ {F} ^ {2}.
$$

$$
l o s s (Y, Z) := \| Y - Z \| _ {F} ^ {2} - \| Y - \widehat {Y} \| _ {F} ^ {2}.
$$

$$
\min  _ {U \in \mathbb {R} ^ {m \times n}, \operatorname {r a n k} (U) \leq d} \max  \left\{\frac {1}{| A |} \operatorname {l o s s} (A, U _ {A}), \frac {1}{| B |} \operatorname {l o s s} (B, U _ {B}) \right\}, \tag {1}
$$

$$
\frac {1}{| A |} \operatorname {l o s s} (A, U _ {A}) = \frac {1}{| B |} \operatorname {l o s s} (B, U _ {B}).
$$

$$
f \left(\left[ \begin{array}{c} A \\ B \end{array} \right] V V ^ {T}\right) = f \left(\left[ \begin{array}{c} A V V ^ {T} \\ B V V ^ {T} \end{array} \right]\right) \leq f (U).
$$

$$
\diamond \operatorname {l o s s} (A, A V V ^ {T}) = \| \widehat {A} \| _ {F} ^ {2} - \sum_ {i = 1} ^ {d} \| A v _ {i} \| ^ {2} = \| \widehat {A} \| _ {F} ^ {2} - \langle A ^ {T} A, V V ^ {T} \rangle
$$

$$
\diamond \| A - A V V ^ {T} \| _ {F} ^ {2} = \| A \| _ {F} ^ {2} - \| A V \| _ {F} ^ {2} = \| A \| _ {F} ^ {2} - \sum_ {i = 1} ^ {d} \| A v _ {i} \| ^ {2}
$$

$$
\operatorname {l o s s} (A, A V V ^ {T}) = \| \widehat {A} \| _ {F} ^ {2} - \| A \| _ {F} ^ {2} + g _ {A} (V), \tag {2}
$$

$$
l o s s (B, B V V ^ {T}) = \| \widehat {B} \| _ {F} ^ {2} - \| B \| _ {F} ^ {2} + g _ {B} (V).
$$

$$
\min  _ {V \in \mathbb {R} ^ {n \times d}, V ^ {T} V = I} f (V) := \max  \left\{\frac {1}{| A |} \operatorname {l o s s} (A, A V V ^ {T}), \frac {1}{| B |} \operatorname {l o s s} (B, B V V ^ {T}) \right\}.
$$

$$
\frac {1}{| A |} \operatorname {l o s s} (A, A W W ^ {T}) > \frac {1}{| B |} \operatorname {l o s s} (B, B W W ^ {T}). \tag {3}
$$

$$
\min  _ {P \in \mathbb {R} ^ {n \times n}, z \in \mathbb {R}} z \tag {4}
$$

$$
s. t. z \geq \frac {1}{m _ {1}} \cdot \left(\| \widehat {A} \| _ {F} ^ {2} - \langle A ^ {\top} A, P \rangle\right)
$$

$$
z \geq \frac {1}{m _ {2}} \cdot \left(\| \widehat {B} \| _ {F} ^ {2} - \langle B ^ {\top} B, P \rangle\right)
$$

$$
\operatorname {T r} (P) \leq d, 0 \preceq P \preceq I
$$

$$
\min  _ {\lambda \in \mathbb {R} ^ {n}, z \in \mathbb {R}} z \tag {5}
$$

$$
\text {s . t .} z \geq \frac {1}{m _ {1}} \left(\| \widehat {A} \| _ {F} ^ {2} - \langle A ^ {\top} A, \sum_ {j = 1} ^ {n} \lambda_ {j} u _ {j} u _ {j} ^ {T} \rangle\right) = \frac {1}{m _ {1}} \left(\| \widehat {A} \| _ {F} ^ {2} - \sum_ {j = 1} ^ {n} \lambda_ {j} \cdot \langle A ^ {\top} A, u _ {j} u _ {j} ^ {T} \rangle\right) \tag {6}
$$

$$
z \geq \frac {1}{m _ {2}} \left(\| \widehat {B} \| _ {F} ^ {2} - \langle B ^ {\top} B, \sum_ {j = 1} ^ {n} \lambda_ {j} u _ {j} u _ {j} ^ {T} \rangle\right) = \frac {1}{m _ {2}} \left(\| \widehat {B} \| _ {F} ^ {2} - \sum_ {j = 1} ^ {n} \lambda_ {j} \cdot \langle B ^ {\top} B, u _ {j} u _ {j} ^ {T} \rangle\right) (7)
$$

$$
\sum_ {i = 1} ^ {n} \lambda_ {i} \leq d \tag {8}
$$

$$
0 \leq \lambda_ {i} \leq 1 \tag {9}
$$

$$
\frac {1}{| A |} (\| \widehat {A} \| _ {F} ^ {2} - \sum_ {i = 1} ^ {n} \bar {\lambda} _ {i} \langle A ^ {T} A, u _ {i} u _ {i} ^ {T} \rangle) = \frac {1}{| B |} (\| \widehat {B} \| _ {F} ^ {2} - \sum_ {i = 1} ^ {n} \bar {\lambda} _ {i} \langle B ^ {T} B, u _ {i} u _ {i} ^ {T} \rangle) = z ^ {*} \leq \widehat {z},
$$

$$
\begin{array}{l} l o s s (A, A P ^ {*}) = \| A - A P ^ {*} \| _ {F} ^ {2} - \| A - \widehat {A} \| _ {F} ^ {2} = \operatorname {T r} \left((A - A P ^ {*}) (A - A P ^ {*}) ^ {\top}\right) - \| A \| _ {F} ^ {2} + \| \widehat {A} \| _ {F} ^ {2} \\ = \operatorname {T r} \left((A - A P ^ {*}) (A - A P ^ {*}) ^ {\top}\right) - \| A \| _ {F} ^ {2} + \| \widehat {A} \| _ {F} ^ {2} = \| \widehat {A} \| _ {F} ^ {2} - 2 \operatorname {T r} (A P ^ {*} A ^ {\top}) + \operatorname {T r} (A P ^ {* 2} A ^ {\top}) \\ = \| \widehat {A} \| _ {F} ^ {2} - \sum_ {i = 1} ^ {n} \left(2 \lambda_ {i} ^ {*} - \lambda_ {i} ^ {* 2}\right) \langle A ^ {T} A, u _ {i} u _ {i} ^ {T} \rangle = \| \widehat {A} \| _ {F} ^ {2} - \sum_ {i = 1} ^ {n} \bar {\lambda} \langle A ^ {T} A, u _ {i} u _ {i} ^ {T} \rangle , \\ \end{array}
$$

$$
\min  _ {U \in \mathbb {R} ^ {m \times n}, \operatorname {r a n k} (U) \leq d} \max  _ {i \in \{1, \dots , k \}} \left\{\frac {1}{| A _ {i} |} \operatorname {l o s s} \left(A _ {i}, U _ {A _ {i}}\right)\right) \Bigg \}, \tag {10}
$$

$$
\exists ? x \in \mathcal {P}: A x \geq b \tag {11}
$$

$$
\exists ? x \in \mathcal {P}: p ^ {\top} A x \geq p ^ {\top} b \tag {12}
$$

$$
\min  z: A x - b + z \cdot \mathbf {1} \geq 0, \text {s . t .} x \in \mathcal {P} \tag {13}
$$

$$
\text {F i n d} x \in \mathcal {P}: p ^ {\top} A x - p ^ {\top} b + z ^ {*} \geq 0 \tag {14}
$$

$$
\min  z: p ^ {\top} A x - p ^ {\top} b + z \geq 0, \text {s . t .} x \in \mathcal {P} \tag {15}
$$

$$
\forall i \in I: A _ {i} x - b _ {i} + z ^ {*} \in [ - \ell , \rho ] \tag {16}
$$

$$
\forall i \notin I: A _ {i} x - b _ {i} + z ^ {*} \in [ - \rho , \ell ] \tag {17}
$$

$$
A x - b + z ^ {*} \cdot \mathbf {1} \geq - \epsilon \tag {18}
$$

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} m ^ {t} \cdot p ^ {t} \leq \sum_ {t = 1} ^ {T} m _ {i} ^ {t} + \eta \sum_ {t = 1} ^ {T} | m _ {i} ^ {t} | + \frac {\log m}{\eta} \\ = \frac {1}{\rho} \sum_ {t = 1} ^ {T} \left(A _ {i} x ^ {t} - b _ {i} + z ^ {*}\right) + \frac {\eta}{\rho} \sum_ {t = 1} ^ {T} \left| A _ {i} x ^ {t} - b _ {i} + z ^ {*} \right| + \frac {\log m}{\eta} \tag {19} \\ \end{array}
$$

$$
\sum_ {t = 1} ^ {T} m ^ {t} \cdot p ^ {t} = \frac {1}{\rho} \sum_ {t = 1} ^ {T} \left(\left(p ^ {t}\right) ^ {\top} \left(A x ^ {t} - b\right) + z ^ {*}\right) \geq 0 \tag {20}
$$

$$
\begin{array}{l} 0 \leq \frac {1 + \eta}{\rho} \sum_ {t = 1} ^ {T} \left(A _ {i} x ^ {t} - b _ {i} + z ^ {*}\right) + \frac {2 \eta}{\rho} \sum_ {t: A _ {i} x ^ {t} - b _ {i} <   0} \left| A _ {i} x ^ {t} - b _ {i} + z ^ {*} \right| + \frac {\log m}{\eta} \\ \leq \frac {1 + \eta}{\rho} T \left(A _ {i} \bar {x} - b _ {i} + z ^ {*}\right) + \frac {2 \eta}{\rho} T \ell + \frac {\log n}{\eta} \\ \end{array}
$$

$$
0 \leq (1 + \eta) \left(A _ {i} \bar {x} - b _ {i} + z ^ {*}\right) + 2 \eta \ell + \frac {\rho \log m}{T \eta} \tag {21}
$$

$$
\begin{array}{l} 0 \leq \frac {1 - \eta}{\rho} \sum_ {t = 1} ^ {T} \left(A _ {i} x ^ {t} - b _ {i} + z ^ {*}\right) + \frac {2 \eta}{\rho} \sum_ {t: A _ {i} x ^ {t} - b _ {i} > 0} \left| A _ {i} x ^ {t} - b _ {i} + z ^ {*} \right| + \frac {\log m}{\eta} \\ \leq \frac {1 - \eta}{\rho} T (A _ {i} \bar {x} - b _ {i}) + \frac {2 \eta}{\rho} T \ell + \frac {\log n}{\eta} \\ \end{array}
$$

$$
0 \leq (1 - \eta) \left(A _ {i} \bar {x} - b _ {i} + z ^ {*}\right) + 2 \eta \ell + \frac {\rho \log m}{T \eta} \tag {22}
$$

$$
2 \eta \ell + \frac {\rho \log m}{T \eta} \leq \frac {\epsilon}{4} + \frac {\epsilon}{4} = \frac {\epsilon}{2} \tag {23}
$$

$$
0 \leq (1 + \eta) \left(A _ {i} \bar {x} - b _ {i} + z ^ {*}\right) + \frac {\epsilon}{2} \Rightarrow A _ {i} \bar {x} - b _ {i} + z ^ {*} \geq - \frac {\epsilon}{2} \tag {24}
$$

$$
0 \leq (1 - \eta) \left(A _ {i} \bar {x} - b _ {i} + z ^ {*}\right) + \frac {\epsilon}{2} \Rightarrow A _ {i} \bar {x} - b _ {i} + z ^ {*} \geq - \epsilon \tag {25}
$$

$$
\min  _ {P \in \mathcal {P}, z \in \mathbb {R}} z \text {s . t .} \tag {26}
$$

$$
z \geq \alpha - \frac {1}{m _ {1}} \langle A ^ {\top} A, P \rangle \tag {27}
$$

$$
z \geq \beta - \frac {1}{m _ {2}} \langle B ^ {\top} B, P \rangle \tag {28}
$$

$$
\mathcal {P} = \{M \in \mathbb {R} ^ {n \times n}: 0 \preceq M \preceq I, \operatorname {t r} (M) \leq d \} \tag {29}
$$

$$
z _ {1} = \alpha - \frac {1}{m _ {1}} \langle A ^ {\top} A, P \rangle ,
$$

$$
z _ {2} = \beta - \frac {1}{m _ {2}} \langle B ^ {\top} B, P \rangle ,
$$

$$
P \in \mathcal {P} = \left\{M \in \mathbb {R} ^ {n \times n}: 0 \preceq M \preceq I, \operatorname {T r} (M) \leq d \right\}
$$

$$
0 \leq \alpha - \frac {1}{m _ {1}} \langle A ^ {\top} A, P \rangle \leq 1, \forall P \in \mathcal {P} \tag {30}
$$

$$
\left\| A _ {i} - \left(U _ {A}\right) _ {i} \right\| ^ {2} = \left\| A _ {i} - c _ {i} V ^ {T} \right\| ^ {2} = A _ {i} A _ {i} ^ {T} - 2 A _ {i} V c _ {i} ^ {T} + c _ {i} c _ {i} ^ {T}
$$

$$
l o s s (A, U _ {A}) = \| A - U _ {A} \| _ {F} ^ {2} - \| A - \widehat {A} \| _ {F} ^ {2} = \sum \| A _ {i} - (U _ {A}) _ {i} \| ^ {2} - \| A - \widehat {A} \| _ {F} ^ {2}
$$

$$
l o s s (A, A V V ^ {T}) = \| A - A V V ^ {T} \| _ {F} ^ {2} - \| A - \widehat {A} \| _ {F} ^ {2} = \sum \| A _ {i} - A _ {i} V V ^ {T} \| ^ {2} - \| A - \widehat {A} \| _ {F} ^ {2}
$$

$$
\begin{array}{l} f (\left[ \begin{array}{c} A \\ B \end{array} \right] V V ^ {T}) = \max  \big (\frac {1}{| A |} l o s s (A, A V V ^ {T}), \frac {1}{| B |} l o s s (B, B V V ^ {T}) \big) \\ \leq \max  \left(\frac {1}{| A |} \operatorname {l o s s} (A, U _ {A}), \frac {1}{| B |} \operatorname {l o s s} (B, U _ {B})\right) \\ = f (U) \\ \end{array}
$$

$$
\begin{array}{l} \operatorname {l o s s} (A, A V V ^ {T}) = \| A - A V V ^ {T} \| _ {F} ^ {2} - \| A - A W _ {A} W _ {A} ^ {T} \| _ {F} ^ {2} \\ = \sum_ {i} \| A _ {i} - A _ {i} V V ^ {T} \| ^ {2} - \| A _ {i} - A _ {i} W _ {A} W _ {A} ^ {T} \| ^ {2} \\ = \sum_ {i} A _ {i} A _ {i} ^ {T} - A _ {i} V V ^ {T} A _ {i} ^ {T} - \left(\sum_ {i} A _ {i} A _ {i} ^ {T} - \sum_ {i} A _ {i} W _ {A} W _ {A} ^ {T}\right) \\ = \sum_ {i} A _ {i} W _ {A} W _ {A} ^ {T} A _ {i} ^ {T} - \sum_ {i} A _ {i} V V ^ {T} A _ {i} ^ {T} \\ \end{array}
$$

$$
\begin{array}{l} \sum_ {i} A _ {i} W _ {A} W _ {A} ^ {T} A _ {i} ^ {T} = \sum_ {i} \| A _ {i} W _ {A} \| ^ {2} = \| A W _ {A} \| _ {F} ^ {2} = \| A W _ {A} W _ {A} ^ {T} \| _ {F} ^ {2} = \| \widehat {A} \| _ {F} ^ {2} \\ \sum_ {i} A _ {i} V V ^ {T} A _ {i} ^ {T} = \sum_ {i} \| A _ {i} V \| ^ {2} = \| A V \| _ {F} ^ {2} = \sum_ {i} \| A v _ {i} \| ^ {2} \\ \sum_ {i} A _ {i} V V ^ {T} A _ {i} ^ {T} = \sum_ {i} \| A _ {i} V \| ^ {2} = \| A V \| _ {F} ^ {2} = \operatorname {T r} (V ^ {T} A ^ {T} A V) = \operatorname {T r} (V V ^ {T} A ^ {T} A) = \langle A ^ {T} A, V V ^ {T} \rangle \\ \end{array}
$$

$$
\begin{array}{l} \| A - A V V ^ {T} \| _ {F} ^ {2} = \sum_ {i} \| A _ {i} - A _ {i} V V ^ {T} \| ^ {2} = \sum_ {i} A _ {i} A _ {i} ^ {T} - \sum_ {i} A _ {i} V V ^ {T} A _ {i} ^ {T} \\ = \| A \| _ {F} ^ {2} - \sum_ {i} \| A v _ {i} \| ^ {2} = \| A \| _ {F} ^ {2} - \| A V \| _ {F} ^ {2} \\ \end{array}
$$

$$
\begin{array}{l} \left\| A w \right\| ^ {2} - \left\| A u _ {1} \right\| ^ {2} = \left\| A \left(\sqrt {1 - \epsilon^ {2}} u _ {1} + \epsilon v _ {1}\right) \right\| ^ {2} - \left\| A u _ {1} \right\| ^ {2} \\ = \left(\sqrt {1 - \epsilon^ {2}} u _ {1} ^ {T} + \epsilon v _ {1} ^ {T}\right) A ^ {T} A \left(\sqrt {1 - \epsilon^ {2}} u _ {1} + \epsilon v _ {1}\right) - \| A u _ {1} \| ^ {2} \\ = (1 - \epsilon^ {2}) u _ {1} ^ {T} A ^ {T} A u _ {1} + \epsilon^ {2} v _ {1} ^ {T} A ^ {T} A v _ {1} + 2 \sqrt {1 - \epsilon^ {2}} \epsilon u _ {1} ^ {T} A ^ {T} A v _ {1} - \| A u _ {1} \| ^ {2} \\ = \left(1 - \epsilon^ {2}\right) \| A u _ {1} \| ^ {2} + \epsilon^ {2} \lambda_ {1} + 2 \epsilon \sqrt {1 - \epsilon^ {2}} u _ {1} ^ {T} A ^ {T} A v _ {1} - \| A u _ {1} \| ^ {2} \\ = \epsilon^ {2} \left(\lambda_ {1} - \| A u _ {1} \| ^ {2}\right) + 2 \epsilon \sqrt {1 - \epsilon^ {2}} u _ {1} ^ {T} A ^ {T} A v _ {1} \\ \end{array}
$$

$$
\left\| A w \right\| ^ {2} - \left\| A u _ {1} \right\| ^ {2} = \epsilon^ {2} \left(\lambda_ {1} - \left\| A u _ {1} \right\| ^ {2}\right) > 0
$$

$$
\begin{array}{l} w = \left(\sqrt {1 - \epsilon^ {2}} - \frac {\epsilon \sqrt {1 - a ^ {2}}}{a}\right) z _ {1} + \frac {\epsilon}{a} (\sqrt {1 - a ^ {2}} z _ {1} + a z _ {2}) = \left(\sqrt {1 - \epsilon^ {2}} - \frac {\epsilon \sqrt {1 - a ^ {2}}}{a}\right) z _ {1} + \frac {\epsilon}{a} v _ {1}. \\ \| A w \| ^ {2} \\ = \left(\sqrt {1 - \epsilon^ {2}} - \frac {\epsilon \sqrt {1 - a ^ {2}}}{a}\right) ^ {2} \| A z _ {1} \| ^ {2} + \frac {\epsilon^ {2}}{a ^ {2}} \| A v _ {1} \| ^ {2} + 2 \frac {\epsilon}{a} \left(\sqrt {1 - \epsilon^ {2}} - \frac {\epsilon \sqrt {1 - a ^ {2}}}{a}\right) z _ {1} ^ {T} A ^ {T} A v _ {1} \\ = \left(1 - \epsilon^ {2} + \frac {\epsilon^ {2} (1 - a ^ {2})}{a ^ {2}} - 2 \frac {\epsilon \sqrt {(1 - \epsilon^ {2}) (1 - a ^ {2})}}{a}\right) \| A z _ {1} \| ^ {2} + \frac {\epsilon^ {2}}{a ^ {2}} \lambda_ {1} \\ + 2 \frac {\epsilon}{a} \left(\sqrt {1 - \epsilon^ {2}} - \frac {\epsilon \sqrt {1 - a ^ {2}}}{a}\right) \lambda_ {1} z _ {1} ^ {T} v _ {1} = \left(1 - 2 \epsilon^ {2} + \frac {\epsilon^ {2}}{a ^ {2}} - 2 \frac {\epsilon \sqrt {(1 - \epsilon^ {2}) (1 - a ^ {2})}}{a}\right) \| A z _ {1} \| ^ {2} \\ + \left(\frac {\epsilon^ {2}}{a ^ {2}} + 2 \frac {\epsilon \sqrt {(1 - \epsilon^ {2}) (1 - a ^ {2})}}{a} - 2 \frac {\epsilon^ {2} (1 - a ^ {2})}{a ^ {2}}\right) \lambda_ {1} \\ = \| A z _ {1} \| ^ {2} + \left(\lambda_ {1} - \| A z _ {1} \| ^ {2}\right) \left(2 \frac {\epsilon \sqrt {(1 - \epsilon^ {2}) (1 - a ^ {2})}}{a} + 2 \epsilon^ {2} - \frac {\epsilon^ {2}}{a ^ {2}}\right) \\ > \left\| A z _ {1} \right\| ^ {2} \\ \end{array}
$$
