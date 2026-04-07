$$
\begin{array}{l} J (X, w, u, b, c) = \\ \sum_ {i = 1} ^ {V} \sum_ {j = 1} ^ {V} f \left(X _ {i j}\right) \left(w _ {i} ^ {T} u _ {j} + b _ {i} + c _ {j} - \log X _ {i j}\right) ^ {2} \tag {1} \\ \end{array}
$$

$$
R (z, \theta) = \frac {1}{n} \sum_ {i = 1} ^ {n} L \left(z _ {i}, \theta\right) \quad \theta^ {*} = \underset {\theta} {\operatorname {a r g m i n}} R (z, \theta) \tag {2}
$$

$$
\tilde {\theta} \approx \theta^ {*} - \frac {1}{n} H _ {\theta^ {*}} ^ {- 1} \sum_ {k \in \delta} \left[ \nabla_ {\theta} L \left(\tilde {z} _ {k}, \theta^ {*}\right) - \nabla_ {\theta} L \left(z _ {k}, \theta^ {*}\right) \right] \tag {3}
$$

$$
g (c, \mathcal {A}, \mathcal {B}, w) = \underset {a \in \mathcal {A}} {\text {m e a n}} \cos \left(w _ {c}, w _ {a}\right) - \underset {b \in \mathcal {B}} {\text {m e a n}} \cos \left(w _ {c}, w _ {b}\right)
$$

$$
B _ {\text {w e a t}} (w) = \frac {\underset {s \in \mathcal {S}} {\text {m e a n}} g (s , \mathcal {A} , \mathcal {B} , w) - \underset {t \in \mathcal {T}} {\text {m e a n}} g (t , \mathcal {A} , \mathcal {B} , w)}{\underset {c \in \mathcal {S} \cup \mathcal {T}} {\text {s t d - d e v}} g (c , \mathcal {A} , \mathcal {B} , w)} \tag {4}
$$

$$
\Delta_ {p} B = B (w) - B (\tilde {w}) \tag {5}
$$

$$
\nabla_ {X} B (w (X)) = \nabla_ {w} B (w) \nabla_ {X} w (X) \tag {6}
$$

$$
B (w (\tilde {X})) \approx B (w (X)) - \nabla_ {X} B (w (X)) \cdot X ^ {(k)}
$$

$$
B (w (X)) - B (w (\tilde {X})) \approx \nabla_ {w} B (w) \nabla_ {X} w (X) \cdot X ^ {(k)}
$$

$$
L (X _ {i}, w) = \sum_ {j = 1} ^ {V} V f (X _ {i j}) \left(w _ {i} ^ {T} u _ {j} + b _ {i} + c _ {j} - \log X _ {i j}\right) ^ {2}
$$

$$
\nabla_ {w} L \left(X _ {i}, w\right) = \left(\underbrace {0 , \dots , 0} _ {V D \text {d i m e n s i o n s}}, \overbrace {\nabla_ {w _ {i}} L \left(X _ {i} , w\right)} ^ {D (i - 1)}, \overbrace {0 , \dots , 0} ^ {D (V - i)}\right) \tag {7}
$$

$$
\sum_ {j = 1} ^ {V} 2 V f \left(X _ {i j}\right) \left(w _ {i} ^ {T} u _ {j} + b _ {i} + c _ {j} - \log X _ {i j}\right) u _ {j}
$$

$$
H _ {w _ {i}} = \frac {1}{V} \nabla_ {w _ {i}} ^ {2} L (X _ {i}, w) = \sum_ {j = 1} ^ {V} 2 f (X _ {i j}) u _ {j} u _ {j} ^ {T}
$$

$$
\tilde {w} _ {i} \approx w _ {i} ^ {*} - \frac {1}{V} H _ {w _ {i}} ^ {- 1} \left[ \nabla_ {w _ {i}} L (\tilde {X} _ {i}, w) - \nabla_ {w _ {i}} L (X _ {i}, w) \right] \tag {8}
$$

$$
\begin{array}{l} \nabla_ {X} B (w (X)) = \nabla_ {w} B (w) \nabla_ {X} w (X) \\ = \sum_ {i = 1} ^ {V} \nabla_ {w _ {i}} B (w) \nabla_ {X} w _ {i} (X) \\ \end{array}
$$

$$
\nabla_ {X} B (w (X)) = \sum_ {i \in \mathcal {U}} \nabla_ {w _ {i}} B (w) \nabla_ {X} w _ {i} (X) \tag {9}
$$

$$
\tilde {w} _ {i} \approx w _ {i} ^ {*} - \frac {1}{V} H _ {w _ {i}} ^ {- 1} \left[ \nabla_ {w _ {i}} L \left(\tilde {X} _ {i}, w ^ {*}\right) - \nabla_ {w _ {i}} L \left(X _ {i}, w ^ {*}\right) \right] \tag {10}
$$

$$
\begin{array}{l} \nabla_ {X} w _ {i} (X) = - \nabla_ {Y} w _ {i} (\tilde {X} (Y)) | _ {Y = 0} \\ \approx - \nabla_ {Y} \left[ w _ {i} ^ {*} - \frac {1}{V} H _ {w _ {i}} ^ {- 1} \left[ \nabla_ {w _ {i}} L \left(\tilde {X} _ {i} (Y), w ^ {*}\right) - \nabla_ {w _ {i}} L \left(X _ {i}, w ^ {*}\right) \right] \right] | _ {Y = 0} \tag {11} \\ \approx \frac {1}{V} H _ {w _ {i}} ^ {- 1} \nabla_ {Y} \nabla_ {w _ {i}} L (\tilde {X} _ {i} (Y), w ^ {*}) | _ {Y = 0} \\ \end{array}
$$

$$
\nabla_ {Y _ {i}} \sum_ {j = 1} ^ {V} 2 V f (X _ {i j} - Y _ {i j}) \left(w _ {i} ^ {T} u _ {j} + b _ {i} + c _ {j} - \log (X _ {i j} - Y _ {i j})\right) u _ {j}
$$

$$
\begin{array}{l} \nabla_ {X} B (w (X)) = \sum_ {i \in \mathcal {U}} \nabla_ {w _ {i}} B (w) \nabla_ {X} w _ {i} (X) \\ \approx \frac {1}{V} \sum_ {i \in \mathcal {U}} \nabla_ {w _ {i}} B (w) H _ {w _ {i}} ^ {- 1} \nabla_ {Y} \nabla_ {w _ {i}} L (\tilde {X} _ {i} (Y), w ^ {*}) | _ {Y = 0} \\ \end{array}
$$

$$
R (z, \theta) = \frac {1}{n} \sum_ {i = 1} ^ {n} L \left(z _ {i}, \theta\right) \quad \theta^ {*} = \underset {\theta} {\operatorname {a r g m i n}} R (z, \theta) \tag {12}
$$

$$
\tilde {\theta} (\varepsilon) = \underset {\theta} {\operatorname {a r g m i n}} \left\{R (z, \theta) + \varepsilon L \left(\tilde {z} _ {k}, \theta\right) - \varepsilon L \left(z _ {k}, \theta\right) \right\} \tag {13}
$$

$$
0 = \nabla_ {\theta} R (z, \tilde {\theta}) + \varepsilon \nabla_ {\theta} L (\tilde {z} _ {k}, \tilde {\theta}) - \varepsilon \nabla_ {\theta} L (z _ {k}, \tilde {\theta})
$$

$$
\begin{array}{l} 0 \approx \nabla_ {\theta} R (z, \theta^ {*}) + \varepsilon \nabla_ {\theta} L (\tilde {z} _ {k}, \theta^ {*}) - \varepsilon \nabla_ {\theta} L (z _ {k}, \theta^ {*}) \\ + \left[ \nabla_ {\theta} ^ {2} R (z, \theta^ {*}) + \varepsilon \nabla_ {\theta} ^ {2} L (\tilde {z} _ {k}, \theta^ {*}) - \varepsilon \nabla_ {\theta} ^ {2} L (z _ {k}, \theta^ {*}) \right] (\tilde {\theta} - \theta^ {*}) \\ \end{array}
$$

$$
\tilde {\theta} - \theta^ {*} \approx \left(\frac {- 1}{n}\right) H _ {\theta^ {*}} ^ {- 1} \left[ \nabla_ {\theta} L \left(\tilde {z} _ {k}, \theta^ {*}\right) - \nabla_ {\theta} L \left(z _ {k}, \theta^ {*}\right) \right] \tag {14}
$$

$$
\tilde {\theta} - \theta^ {*} \approx \left(\frac {- 1}{n}\right) H _ {\theta^ {*}} ^ {- 1} \sum_ {k \in \delta} \left[ \nabla_ {\theta} L \left(\tilde {z} _ {k}, \theta^ {*}\right) - \nabla_ {\theta} L \left(z _ {k}, \theta^ {*}\right) \right] \tag {15}
$$
