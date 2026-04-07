$$
\Pr \left\{\widehat {Y} = 1 \mid A = 0, Y = y \right\} = \Pr \left\{\widehat {Y} = 1 \mid A = 1, Y = y \right\}, \quad y \in \{0, 1 \}
$$

$$
\gamma_ {a} (\widehat {Y}) \stackrel {\text {d e f}} {=} \left(\Pr \left\{\widehat {Y} = 1 \mid A = a, Y = 0 \right\}, \Pr \left\{\widehat {Y} = 1 \mid A = a, Y = 1 \right\}\right). \tag {4.1}
$$

$$
P _ {a} (\widehat {Y}) \stackrel {\text {d e f}} {=} \operatorname {c o n v h u l l} \left\{(0, 0), \gamma_ {a} (\widehat {Y}), \gamma_ {a} (1 - \widehat {Y}), (1, 1) \right\} \tag {4.2}
$$

$$
\min  _ {\widetilde {Y}} \mathbb {E} \ell (\widetilde {Y}, Y) \tag {4.3}
$$

$$
\text {s . t .} \quad \forall a \in \{0, 1 \}: \gamma_ {a} (\widetilde {Y}) \in P _ {a} (\widehat {Y}) \quad \left(\text {d e r i v e d}\right)
$$

$$
\gamma_ {0} (\widetilde {Y}) = \gamma_ {1} (\widetilde {Y}) \quad \text {(e q u a l i z e d o d d s)}
$$

$$
\mathbb {E} \left[ \ell (\widetilde {Y}, Y) \right] = \sum_ {y, y ^ {\prime} \in \{0, 1 \}} \ell (y, y ^ {\prime}) \Pr \left\{\widetilde {Y} = y ^ {\prime}, Y = y \right\}.
$$

$$
\begin{array}{l} \Pr \left\{\widetilde {Y} = y ^ {\prime}, Y = y \right\} = \Pr \left\{\widetilde {Y} = y ^ {\prime}, Y = y \mid \widetilde {Y} = \widehat {Y} \right\} \Pr \left\{\widetilde {Y} = \widehat {Y} \right\} \\ + \Pr \left\{\widetilde {Y} = y ^ {\prime}, Y = y \mid \widetilde {Y} \neq \widehat {Y} \right\} \Pr \left\{\widetilde {Y} \neq \widehat {Y} \right\} \\ = \Pr \left\{\widehat {Y} = y ^ {\prime}, Y = y \right\} \Pr \left\{\widetilde {Y} = \widehat {Y} \right\} + \Pr \left\{\widehat {Y} = 1 - y ^ {\prime}, Y = y \right\} \Pr \left\{\widetilde {Y} \neq \widehat {Y} \right\}. \\ \end{array}
$$

$$
C _ {a} (t) \stackrel {{\mathrm {d e f}}} {{=}} \left(\Pr \left\{\widehat {R} > t \mid A = a, Y = 0 \right\}, \Pr \left\{\widehat {R} > t \mid A = a, Y = 1 \right\}\right).
$$

$$
D _ {a} \stackrel {\text {d e f}} {=} \operatorname {c o n v h u l l} \left\{C _ {a} (t): t \in [ 0, 1 ] \right\} \tag {4.4}
$$

$$
\widetilde {Y} = \mathbb {I} \left\{R > T _ {a} \right\},
$$

$$
\min  _ {\forall a: \gamma \in D _ {a}} \gamma_ {0} \ell (1, 0) + (1 - \gamma_ {1}) \ell (0, 1) \tag {4.5}
$$

$$
d _ {\mathrm {K}} \left(R, R ^ {\prime}\right) \stackrel {\text {d e f}} {=} \max  _ {a, y \in \{0, 1 \}} \sup  _ {t \in [ 0, 1 ]} \left| \Pr \left\{R > t \mid A = a, Y = y \right\} - \Pr \left\{R ^ {\prime} > t \mid A = a, Y = y \right\} \right|. \tag {5.1}
$$

$$
\mathbb {E} \ell (\widehat {Y}, Y) \leqslant \mathbb {E} \ell (Y ^ {*}, Y) + 2 \sqrt {2} \cdot d _ {\mathrm {K}} (\widehat {R}, R ^ {*}),
$$

$$
\left\| p ^ {*} - q _ {a} \right\| _ {2} \leqslant \sqrt {2} \cdot d _ {\mathrm {K}} (\widehat {R}, R ^ {*}).
$$

$$
\left\| p ^ {*} - q \right\| _ {2} \leqslant 2 \cdot d _ {\mathrm {K}} (\widehat {R}, R ^ {*}).
$$

$$
\left\| p ^ {*} - q \right\| _ {2} ^ {2} \leqslant \left\| p ^ {*} - q _ {0} \right\| _ {2} ^ {2} + \left\| p ^ {*} - q _ {0} \right\| _ {2} ^ {2} \leqslant 4 \cdot d _ {\mathrm {K}} \left(\widehat {R}, R ^ {*}\right) ^ {2}.
$$

$$
\mathbb {E} \ell (\widehat {Y}, Y) - \mathbb {E} \ell (Y ^ {*}, Y) = \langle v, q - p ^ {*} \rangle \leqslant \| v \| _ {2} \cdot \| q - p ^ {*} \| _ {2} \leqslant 2 \sqrt {2} \cdot d _ {\mathrm {K}} (\widehat {R}, R ^ {*}).
$$

$$
\Pr \left\{Y = 1 \mid R = t, A = a \right\} = \Pr \left\{Y = 1 \mid R = t, A = a ^ {\prime} \right\}.
$$

$$
\begin{array}{l} \Pr \left\{Y = y \mid X _ {1} = x _ {1}, X _ {2} = x _ {2} \right\} \propto \Pr \left\{A = x _ {1} \right\} \Pr \left\{Y = y \mid A = x _ {1} \right\} \Pr \left\{X _ {2} = x _ {2} \mid Y = y \right\} \\ \propto \exp (2 y (x _ {1} + x _ {2})). \\ \end{array}
$$
