$$
\left| \ell (y, u) - \ell \left(y ^ {\prime}, u ^ {\prime}\right) \right| \leq \left| y - y ^ {\prime} \right| + \left| u - u ^ {\prime} \right| \quad \text {f o r a l l} y, y ^ {\prime}, u, u ^ {\prime}.
$$

$$
\min  _ {f \in \mathcal {F}} \mathbb {E} [ \ell (Y, f (X)) ] \quad \text {s u c h t h a t} \forall a \in \mathcal {A}, z \in [ 0, 1 ];
$$

$$
\left| \mathbb {P} [ f (X) \geq z \mid A = a ] - \mathbb {P} [ f (X) \geq z ] \right| \leq \varepsilon_ {a}. \tag {1}
$$

$$
\min  _ {f \in \mathcal {F}} \mathbb {E} \left[ \ell (Y, f (X)) \right]
$$

$$
\begin{array}{l} \operatorname {l o s s} (f) := \mathbb {E} [ \ell (Y, f (X)) ], \\ \gamma_ {a, z} (f) := \mathbb {P} [ f (X) \geq z \mid A = a ] - \mathbb {P} [ f (X) \geq z ]. \\ \gamma_ {a} ^ {\mathrm {B G L}} (f) := \mathbb {E} \left[ \ell (Y, f (X)) \mid A = a \right]. \\ \end{array}
$$

$$
\min  _ {Q \in \Delta (\mathcal {F})} \operatorname {l o s s} (Q) \text {s . t .} | \gamma_ {a, z} (Q) | \leq \varepsilon_ {a} \forall a \in \mathcal {A}, z \in [ 0, 1 ], \tag {3}
$$

$$
\min  _ {Q \in \Delta (\mathcal {F})} \operatorname {l o s s} (Q) \text {s . t .} \gamma_ {a} ^ {\mathrm {B G L}} (Q) \leq \zeta_ {a} \forall a \in \mathcal {A}. \tag {4}
$$

$$
\ell_ {\alpha} (y, u) := \ell \left(\underline {{y}}, \lfloor u \rfloor_ {\alpha} + \frac {\alpha}{2}\right) \tag {5}
$$

$$
\left| \ell (y, u) - \ell_ {\alpha} (y, u) \right| \leq \alpha . \tag {6}
$$

$$
\begin{array}{l} \gamma_ {a, z} (\underline {{f}}) = \mathbb {P} [ \underline {{f}} (X) \geq z \mid A = a ] - \mathbb {P} [ \underline {{f}} (X) \geq z ] \\ = \mathbb {P} [ f (X) \geq \bar {z} \mid A = a ] - \mathbb {P} [ f (X) \geq \bar {z} ], \tag {7} \\ \end{array}
$$

$$
\min  _ {Q \in \Delta (\mathfrak {F})} \operatorname {l o s s} _ {\alpha} (Q) \text {s . t .} | \gamma_ {a, z} (Q) | \leq \varepsilon_ {a} \forall a \in \mathcal {A}, z \in \mathcal {Z}. \tag {8}
$$

$$
c (y, z) := N \left(\ell \left(y, z + \frac {\alpha}{2}\right) - \ell \left(y, z - \frac {\alpha}{2}\right)\right), \tag {9}
$$

$$
\gamma_ {a, z} (h _ {f}) = \mathbb {E} [ h _ {f} (X, z) | A = a ] - \mathbb {E} [ h _ {f} (X, z) ].
$$

$$
\widehat {\operatorname {c o s t}} \left(h _ {f}\right) = \widehat {\mathbb {E}} \left[ \mathbb {E} _ {Z} [ c (Y, Z) h _ {f} (X, Z) ] \right] \tag {10}
$$

$$
\widehat {\gamma} _ {a, z} (h _ {f}) = \widehat {\mathbb {E}} \left[ h _ {f} (X, z) \mid A = a \right] - \widehat {\mathbb {E}} \left[ h _ {f} (X, z) \right].
$$

$$
\min  _ {Q \in \Delta (\mathcal {H})} \widehat {\operatorname {c o s t}} (Q) \text {s . t .} \left| \widehat {\gamma} _ {a, z} (Q) \right| \leq \widehat {\varepsilon} _ {a} \forall a \in \mathcal {A}, z \in \mathcal {Z}. \tag {11}
$$

$$
\begin{array}{l} L (Q, \boldsymbol {\lambda}) = \widehat {\operatorname {c o s t}} (Q) + \sum \left[ \lambda_ {a, z} ^ {+} \left(\widehat {\gamma} _ {a, z} (Q) - \widehat {\varepsilon} _ {a}\right) \right. \\ \left. + \lambda_ {a, z} ^ {-} \left(- \widehat {\gamma} _ {a, z} (Q) - \widehat {\varepsilon} _ {a}\right) \right]. \\ \end{array}
$$

$$
\begin{array}{l} \operatorname {B E S T} _ {h} (\boldsymbol {\lambda}) := \arg \min  _ {h _ {f} \in \mathcal {H}} L (h _ {f}, \boldsymbol {\lambda}) \\ \operatorname {B E S T} _ {\boldsymbol {\lambda}} (Q) := \arg \max  _ {\boldsymbol {\lambda} \geq 0, \| \boldsymbol {\lambda} \| _ {1} \leq B} L (Q, \boldsymbol {\lambda}) \\ \end{array}
$$

$$
/ / C o m p u t e \lambda_ {t} f r o m \theta_ {t} a n d f i n d t h e b e s t r o p s h e s o n g h _ {t}
$$

$$
/ / C a l c u l a t e \text {c h e t e r n a c t}
$$

$$
/ / C h e c t \left(\widehat {Q} _ {t}, \widehat {\lambda} _ {t}\right)
$$

$$
/ / A p p l y \text {t h e}
$$

$$
\begin{array}{l} \left. \operatorname {l o s s} (\widehat {Q}) \leq \operatorname {l o s s} (Q ^ {\star}) + \widetilde {O} (n ^ {- \beta}) \right. \\ \left| \gamma_ {a, z} (\widehat {Q}) \right| \leq \varepsilon_ {a} + \widetilde {O} \left(n _ {a} ^ {- \beta}\right) \quad f o r a l l a \in \mathcal {A}, z \in [ 0, 1 ]. \\ \end{array}
$$

$$
\begin{array}{l} \widehat {\operatorname {c o s t}} (h) + \sum_ {a, z} \lambda_ {a, z} \widehat {\gamma} _ {a, z} (h) \tag {12} \\ = \widehat {\mathbb {E}} \left[ \mathbb {E} _ {Z} \left[ \underbrace {c (\underline {{Y}} , Z) + \frac {N \lambda_ {A , Z}}{p _ {A}} - \sum_ {a} N \lambda_ {a , Z}} _ {c _ {\lambda} (Y, A, Z)}\right) h (X, Z) \right]. \\ \end{array}
$$

$$
X _ {i, z} ^ {\prime} = \left(X _ {i}, z\right), \quad C _ {i, z} = c _ {\boldsymbol {\lambda}} \left(Y _ {i}, A _ {i}, z\right).
$$

$$
g _ {\lambda} (\tilde {Y}, A, f (X)) = \frac {1}{N} \sum_ {z \in \mathbb {Z}} c _ {\lambda} (\tilde {Y}, A, z) h _ {f} (X, z)
$$

$$
\min  _ {f \in \mathcal {F}} \sum_ {i = 1} ^ {n} (U _ {i} - f (X _ {i})) ^ {2}.
$$

$$
\min  _ {Q \in \Delta (\mathcal {F})} \widehat {\operatorname {l o s s}} (Q) \text {s . t .} \widehat {\gamma} _ {a} ^ {\mathrm {B G L}} (Q) \leq \widehat {\zeta} _ {a} \forall a \in \mathcal {A}. \tag {13}
$$

$$
L ^ {\mathrm {B G L}} (Q, \boldsymbol {\lambda}) = \widehat {\operatorname {l o s s}} (Q) + \sum_ {a} \lambda_ {a} \left(\widehat {\gamma} _ {a} ^ {\mathrm {B G L}} (Q) - \widehat {\zeta} _ {a}\right).
$$

$$
\min  _ {Q \in \Delta} \max  _ {\lambda \geq \mathbf {0}, \| \lambda \| _ {1} \leq B} L ^ {\mathrm {B G L}} (Q, \lambda). \tag {14}
$$

$$
\min  _ {f \in \mathcal {F}} \left[ \widehat {\operatorname {l o s s}} (f) + \sum_ {a} \lambda_ {a} \widehat {\gamma} _ {a} ^ {\mathrm {B G L}} (f) \right].
$$

$$
\min _ {f \in \mathcal {F}} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \ell (Y _ {i}, f (X _ {i})) + \sum_ {a} \frac {\lambda_ {a}}{n _ {a}} \sum_ {i: A _ {i} = a} \ell (Y _ {i}, f (X _ {i})) \right],
$$

$$
\left. \operatorname {l o s s} (\widehat {Q}) \leq \operatorname {l o s s} (Q ^ {\star}) + \widetilde {O} \left(n ^ {- \omega}\right) \right.
$$

$$
\gamma_ {a} ^ {\mathrm {B G L}} (\widehat {Q}) \leq \zeta_ {a} + \widetilde {O} \left(n _ {a} ^ {- \omega}\right) \quad f o r a l l a \in \mathcal {A}.
$$

$$
\begin{array}{l} \ell_ {\alpha} (y, u) = \ell \left(\underline {{y}}, \lfloor u \rfloor_ {\alpha} + \frac {\alpha}{2}\right) \\ = \ell (\underline {{y}}, \frac {\alpha}{2}) + \sum_ {z \in \mathcal {Z}} \left[ \underbrace {\ell \left(\underline {{y}} , z + \frac {\alpha}{2}\right) - \ell \left(\underline {{y}} , z - \frac {\alpha}{2}\right)} _ {c (\underline {{y}}, z) / N} \right] \mathbf {1} \{u \geq z \}. \\ \end{array}
$$

$$
\ell_ {\alpha} (y, \underline {{f}} (x)) = \ell (\underline {{y}}, \frac {\alpha}{2}) + \frac {1}{N} \sum_ {z \in \mathcal {Z}} c (\underline {{y}}, z) h _ {f} (x, z). \tag {15}
$$

$$
\begin{array}{l} \gamma_ {a, z} (\underline {{f}}) = \mathbb {P} [ f (X) \geq z \mid A = a ] - \mathbb {P} [ f (X) \geq z ] \\ = \mathbb {E} \left[ h _ {f} (X, z) \mid A = a \right] - \mathbb {E} \left[ h _ {f} (X, z) \right] \tag {16} \\ = \gamma_ {a, z} (h _ {f}), \\ \end{array}
$$

$$
\begin{array}{l} \widehat {\operatorname {c o s t}} (\widehat {Q}) \leq \widehat {\operatorname {c o s t}} (Q) + 2 \nu \\ \left| \widehat {\gamma} _ {a, z} (\widehat {Q}) \right| \leq \widehat {\varepsilon} _ {a} + \frac {2 + 2 \nu}{B} \quad f o r a l l a \in \mathcal {A}, z \in \mathcal {Z}. \\ \end{array}
$$

$$
R _ {n} (\mathcal {G}) := \sup  _ {u _ {1}, \dots , u _ {n} \in \mathcal {U}} \mathbb {E} _ {\sigma} \left[ \sup  _ {g \in \mathcal {G}} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} g \left(u _ {i}\right) \right| \right], \tag {17}
$$

$$
\left| \widehat {\mathbb {E}} [ \varphi (S, g (U)) ] - \mathbb {E} [ \varphi (S, g (U)) ] \right| \leq 4 R _ {n} (\mathcal {G}) + \frac {2}{\sqrt {n}} + \sqrt {\frac {2 \ln (2 / \delta)}{n}},
$$

$$
\left| \widehat {\mathbb {E}} [ \varphi (S, g (U)) ] - \mathbb {E} [ \varphi (S, g (U)) ] \right| = \left| \widehat {\mathbb {E}} [ \varphi_ {g} ] - \mathbb {E} [ \varphi_ {g} ] \right| \leq 2 R _ {n} (\Phi) + \sqrt {\frac {2 \ln (2 / \delta)}{n}}. \tag {18}
$$

$$
\begin{array}{l} \mathbb {E} _ {\sigma} \left[ \sup  _ {g \in \mathcal {G}} \left| \sum_ {i = 1} ^ {n} \sigma_ {i} \varphi (s _ {i}, g (u _ {i})) \right| \right] \leq \mathbb {E} _ {\sigma} \left[ \sup  _ {g \in \mathcal {G}} \left| \sum_ {i = 1} ^ {n} \sigma_ {i} (\varphi (s _ {i}, g (u _ {i})) - \varphi (s _ {i}, 0)) \right| \right] + \sqrt {n} \\ \leq 2 \mathbb {E} _ {\sigma} \left[ \sup  _ {g \in \mathcal {G}} \left| \sum_ {i = 1} ^ {n} \sigma_ {i} g (u _ {i}) \right| \right] + \sqrt {n} \\ \end{array}
$$

$$
R _ {n} (\Phi) \leq 2 R _ {n} (\mathcal {G}) + \frac {1}{\sqrt {n}}.
$$

$$
\min  _ {Q \in \Delta (\mathcal {H})} \operatorname {c o s t} (Q) \text {s . t .} | \gamma_ {a, z} (Q) | \leq \varepsilon_ {a} \forall a \in \mathcal {A}, z \in \mathcal {Z}. \tag {19}
$$

$$
\widehat {\operatorname {c o s t}} _ {z} (h) = \widehat {\mathbb {E}} \big [ c (\underline {{Y}}, z) h (X, z) \big ] \quad \text {a n d} \quad \operatorname {c o s t} _ {z} (h) = \mathbb {E} \big [ c (\underline {{Y}}, z) h (X, z) \big ].
$$

$$
\left| \widehat {\operatorname {c o s t}} _ {z} (h) - \operatorname {c o s t} _ {z} (h) \right| \leq 2 R _ {n} (\mathcal {H}) + \frac {2}{\sqrt {n}} + \sqrt {\frac {2 \ln (4 N / \delta)}{n}} = \widetilde {O} (n ^ {- \beta}),
$$

$$
\left| \widehat {\operatorname {c o s t}} (Q) - \operatorname {c o s t} (Q) \right| = \widetilde {O} \left(n ^ {- \beta}\right). \tag {20}
$$

$$
\left| \widehat {\gamma} _ {a, z} (h) - \gamma_ {a, z} (h) \right| \leq 2 R _ {n _ {a}} (\mathcal {H}) + \frac {2}{\sqrt {n _ {a}}} + \sqrt {\frac {2 \ln (4 | \mathcal {A} | N / \delta)}{n _ {a}}}.
$$

$$
\left| \widehat {\gamma} _ {a, z} (Q) - \gamma_ {a, z} (Q) \right| \leq 2 R _ {n _ {a}} (\mathcal {H}) + \frac {2}{\sqrt {n _ {a}}} + \sqrt {\frac {2 \ln (4 | \mathcal {A} | N / \delta)}{n _ {a}}}. \tag {21}
$$

$$
\widehat {\operatorname {c o s t}} (\widehat {Q}) \leq \widehat {\operatorname {c o s t}} (Q) + O \left(n ^ {- \beta}\right) \tag {22}
$$

$$
\left| \widehat {\gamma} _ {a, z} (\widehat {Q}) \right| \leq \widehat {\varepsilon} _ {a} + O \left(n ^ {- \beta}\right) \quad \text {f o r a l l} a \in \mathcal {A}, z \in \mathcal {Z}. \tag {23}
$$

$$
\operatorname {c o s t} (\widehat {Q}) \leq \operatorname {c o s t} (Q) + \widetilde {O} \left(n ^ {- \beta}\right). \tag {24}
$$

$$
\left| \gamma_ {a, z} (\widehat {Q}) \right| \leq \varepsilon_ {a} + \widetilde {O} \left(n _ {a} ^ {- \beta}\right) \quad \text {f o r a l l} a \in \mathcal {A}, z \in \mathcal {Z}. \tag {25}
$$

$$
\left. \operatorname {l o s s} _ {\alpha} (\widehat {Q}) \leq \operatorname {l o s s} _ {\alpha} (Q) + \widetilde {O} \left(n ^ {- \beta}\right) \right. \tag {26}
$$

$$
\left| \gamma_ {a, z} (\widehat {Q}) \right| \leq \varepsilon_ {a} + \widetilde {O} \left(n _ {a} ^ {- \beta}\right) \quad \text {f o r a l l} a \in \mathcal {A}, z \in [ 0, 1 ], \tag {27}
$$

$$
\left. \left. \operatorname {l o s s} (\widehat {Q}) \leq \operatorname {l o s s} (Q ^ {*}) + \alpha + \widetilde {O} \left(n ^ {- \beta}\right) = \operatorname {l o s s} (Q ^ {*}) + \widetilde {O} \left(n ^ {- \beta}\right), \right. \right. \tag {28}
$$

$$
\widehat {\operatorname {l o s s}} (\widehat {Q}) \leq \widehat {\operatorname {l o s s}} (Q) + 2 \nu \tag {29}
$$

$$
\widehat {\gamma} _ {a} ^ {\mathrm {B G L}} (\widehat {Q}) \leq \widehat {\zeta} _ {a} + \frac {1 + 2 \nu}{B} \quad \text {f o r a l l} a \in \mathcal {A}, \tag {30}
$$

$$
\left| \widehat {\operatorname {l o s s}} (Q) - \operatorname {l o s s} (Q) \right| \leq 4 R _ {n} (\mathcal {F}) + \frac {2}{\sqrt {n}} + \sqrt {\frac {2 \ln (4 / \delta)}{n}}
$$

$$
\left| \hat {\gamma} _ {a} ^ {\mathrm {B G L}} (Q) - \gamma_ {a} ^ {\mathrm {B G L}} (Q) \right| \leq 4 R _ {n _ {a}} (\mathcal {F}) + \frac {2}{\sqrt {n _ {a}}} + \sqrt {\frac {2 \ln (4 | \mathcal {A} | / \delta)}{n _ {a}}} \quad \text {f o r a \in \mathcal {A}}.
$$

$$
\left| \widehat {\operatorname {l o s s}} (Q) - \operatorname {l o s s} (Q) \right| = \widetilde {O} \left(n ^ {- \omega}\right) \tag {31}
$$

$$
\left| \hat {\gamma} _ {a} ^ {\mathrm {B G L}} (Q) - \gamma_ {a} ^ {\mathrm {B G L}} (Q) \right| \leq C ^ {\prime} n _ {a} ^ {- \omega} \quad \text {f o r a l l} a \in \mathcal {A}. \tag {32}
$$

$$
\left. \operatorname {l o s s} (\widehat {Q}) \leq \operatorname {l o s s} (Q) + 2 \nu + \widetilde {O} \left(n ^ {- \omega}\right) = \operatorname {l o s s} (Q) + \widetilde {O} \left(n ^ {- \omega}\right), \right. \tag {33}
$$

$$
\gamma_ {a} ^ {\mathrm {B G L}} (\widehat {Q}) \leq \widehat {\zeta} _ {a} + \frac {1 + 2 \nu}{B} + \widetilde {O} \left(n _ {a} ^ {- \omega}\right) \leq \zeta_ {a} + \widetilde {O} \left(n _ {a} ^ {- \omega}\right) \quad \text {f o r a l l} a \in \mathcal {A}, \tag {34}
$$

$$
\widehat {\gamma} _ {a} ^ {\mathrm {B G L}} (\widehat {Q}) \leq \widehat {\zeta} _ {a} + \frac {1 + 2 \nu}{B} \quad \text {f o r a l l} a \in \mathcal {A}, \tag {35}
$$

$$
\min  _ {Q \in \Delta} \max  _ {\lambda \geq \mathbf {0}, \| \boldsymbol {\lambda} \| _ {1} \leq B} L (Q, \boldsymbol {\lambda}). \tag {36}
$$

$$
\widehat {\operatorname {c o s t}} (h _ {f}) = \widehat {\mathbb {E}} \left[ \ell_ {\alpha} (Y, \underline {{f}} (X)) - \ell \left(Y, \frac {\alpha}{2}\right) \right],
$$

$$
\widehat {\mathbb {E}} [ h _ {f} (X, z) \mid A = a ] = \widehat {\mathbb {P}} [ f (X) \geq z \mid A = a ]
$$

$$
(a ^ {*}, z ^ {*}) = \underset {(a, z)} {\arg \max } \left[ | \widehat {\gamma} _ {a, z} (Q) | - \widehat {\varepsilon} _ {a} \right]
$$

$$
\left\{ \begin{array}{l l} B \mathbf {e} _ {a ^ {*}, z ^ {*}} ^ {+} & \text {i f} \widehat {\gamma} _ {a ^ {*}, z ^ {*}} (Q) > \widehat {\varepsilon} _ {a ^ {*}} \\ B \mathbf {e} _ {a ^ {*}, z ^ {*}} ^ {-} & \text {i f} \widehat {\gamma} _ {a ^ {*}, z ^ {*}} (Q) <   - \widehat {\varepsilon} _ {a ^ {*}} \\ \mathbf {0} & \text {o t h e r w i s e .} \end{array} \right.
$$

$$
\begin{array}{l} \sum_ {a, z} \lambda_ {a, z} \widehat {\gamma} _ {a, z} (h) = \sum_ {a, z} \lambda_ {a, z} \left(\frac {1}{p _ {a}} \widehat {\mathbb {E}} [ h (X, z) \mathbf {1} \{A = a \} ] - \widehat {\mathbb {E}} [ h (X, z) ]\right) \\ = \widehat {\mathbb {E}} \left[ N \mathbb {E} _ {Z} \left[ \sum_ {a} \lambda_ {a, Z} h (X, Z) \left(\frac {\mathbf {1} \{A = a \}}{p _ {a}} - 1\right) \right] \right] \\ = \widehat {\mathbb {E}} \left[ N \mathbb {E} _ {Z} \left[ \left(\frac {\lambda_ {A , Z}}{p _ {A}} - \sum_ {a} \lambda_ {a, Z}\right) h (X, Z) \right] \right]. \tag {37} \\ \end{array}
$$

$$
\widehat {\operatorname {c o s t}} \left(h _ {f}\right) + \sum_ {a, z} \lambda_ {a, z} \widehat {\gamma} _ {a, z} \left(h _ {f}\right) = \frac {1}{n} \sum_ {i \leq n} \sum_ {z \in \mathbb {Z}} \frac {1}{N} c _ {\boldsymbol {\lambda}} \left(\underline {{Y}} _ {i}, A _ {i}, z\right) h _ {f} \left(X _ {i}, z\right). \tag {38}
$$

$$
g _ {\boldsymbol {\lambda}} (\tilde {y}, a, u) = \sum_ {z \in \mathcal {Z}, z \leq u} \frac {1}{N} c _ {\boldsymbol {\lambda}} (\tilde {y}, a, z).
$$

$$
\begin{array}{l} \sum_ {z \in \mathbb {Z}} \frac {1}{N} c _ {\boldsymbol {\lambda}} (\tilde {Y}, A, z) h _ {f} (X, z) = \sum_ {z \in \mathbb {Z}} \left[ g _ {\boldsymbol {\lambda}} (\tilde {Y}, A, z) - g _ {\boldsymbol {\lambda}} (\tilde {Y}, A, z - \alpha) \right] \mathbf {1} \{f (X) \geq z \} \\ = g _ {\lambda} (\tilde {Y}, A, \lfloor f (X) \rfloor_ {\alpha}) \\ = g _ {\lambda} (\tilde {Y}, A, f (X)). \\ \end{array}
$$

$$
\min  _ {f \in \mathcal {F}} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} g _ {\boldsymbol {\lambda}} \left(\underline {{Y}} _ {i}, A _ {i}, f \left(X _ {i}\right)\right) \right].
$$

$$
U_{i}\in \operatorname *{arg  min}_{u\in [0,1]}g_{\boldsymbol {\lambda}}(\underline{Y}_{i},A_{i},u)
$$

$$
\min  \sum_ {i \leq n} (U _ {i} - f (X _ {i})) ^ {2}.
$$

$$
W _ {i, 1} \frac {\partial}{\partial u} \ell \left(\tilde {Y} _ {i, 1}, U _ {i}\right) + W _ {i, 2} \frac {\partial}{\partial u} \ell \left(\tilde {Y} _ {i, 2}, U _ {i}\right) = 0.
$$

$$
\min  _ {f \in \mathcal {F}} \sum_ {i = 1} ^ {n} \left[ W _ {i, 1} \ell \left(\tilde {Y} _ {i, 1}, f (X _ {i})\right) + W _ {i, 2} \ell \left(\tilde {Y} _ {i, 2}, f (X _ {i})\right) \right].
$$

$$
\min  _ {h \in \mathcal {H}} \sum_ {i = 1} ^ {n} W _ {i} \mathbf {1} \left\{h \left(X _ {i} ^ {\prime}\right) \neq Y _ {i} \right\}. \tag {39}
$$

$$
\min  _ {\beta} \sum_ {i = 1} ^ {n} W _ {i} \max  \left\{0, \frac {\alpha}{2} - \langle \beta , x _ {i} \rangle Y _ {i} \right\}
$$

$$
\min  _ {\beta , t} \sum_ {i} t _ {i}
$$

$$
\text {f o r a l l} i \in [ n ]: \quad t _ {i} \geq 0,
$$

$$
\text {f o r a l l} i \in [ n ]: t _ {i} \geq \frac {\alpha}{2} - Y _ {i} \langle \beta , x _ {i} \rangle ,
$$

$$
f o r \quad j \in [ d ]: - 1 \leq \beta_ {j} \leq 1.
$$
