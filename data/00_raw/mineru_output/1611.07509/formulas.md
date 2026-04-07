$$
P (\mathbf {v}) = \prod_ {V \in \mathbf {V}} P (v | P a (V)), \tag {1}
$$

$$
P (\mathbf {y} | d o (\mathbf {x})) = \prod_ {Y \in \mathbf {Y}} P (y | P a (Y)) \delta_ {\mathbf {X} = \mathbf {x}},
$$

$$
P (y | d o (x)) = \sum_ {\mathbf {V} \backslash \{X, Y \}, Y = y} \prod_ {V \in \mathbf {V} \backslash \{X \}} P (v | P a (V)) \delta_ {X = x}, \tag {2}
$$

$$
T E \left(x _ {2}, x _ {1}\right) = P \left(y \mid d o \left(x _ {2}\right)\right) - P \left(y \mid d o \left(x _ {1}\right)\right).
$$

$$
S E _ {\pi} \left(x _ {2}, x _ {1}\right) = P (y \mid d o \left(x _ {2} \mid_ {\pi}\right)) - P (y \mid d o \left(x _ {1}\right)).
$$

$$
S E _ {\pi_ {d}} \left(c ^ {+}, c ^ {-}\right) = P \left(e ^ {+} \mid d o \left(c ^ {+} \mid_ {\pi_ {d}}\right)\right) - P \left(e ^ {+} \mid d o \left(c ^ {-}\right)\right).
$$

$$
S E _ {\pi_ {i}} \left(c ^ {+}, c ^ {-}\right) = P \left(e ^ {+} \mid d o \left(c ^ {+} \mid_ {\pi_ {i}}\right)\right) - P \left(e ^ {+} \mid d o \left(c ^ {-}\right)\right),
$$

$$
S E _ {\pi} \left(c ^ {+}, c ^ {-}\right) = T E \left(c ^ {+}, c ^ {-}\right) = P \left(e ^ {+} \mid d o \left(c ^ {+}\right)\right) - P \left(e ^ {+} \mid d o \left(c ^ {-}\right)\right).
$$

$$
\begin{array}{l}S E _ {\pi_ {d}} \left(c ^ {+}, c ^ {-}\right) = \sum_ {\mathbf {V} \backslash \{C, E \}} \left(P \left(e ^ {+} \mid c ^ {+}, P a (E) \backslash \{C \}\right) \right.\\\prod_ {V \in \mathbf {V} \backslash \{C, E \}} P (v | P a (V)) \delta_ {C = c ^ {-}}\left. \right) - P \left(e ^ {+} \mid c ^ {-}\right).\end{array}\tag {3}
$$

$$
\begin{array}{l} S E _ {\pi_ {i}} \left(c ^ {+}, c ^ {-}\right) = \sum_ {\mathbf {V} \backslash \{C \}} \left(\prod_ {G \in \mathbf {S} _ {\pi_ {i}}} P (g | c ^ {+}, P a (G) \backslash \{C \}) \right. \\ \prod_ {H \in \mathbf {S} _ {\pi_ {i}}} P (h | c ^ {-}, P a (H) \backslash \{C \}) \prod_ {O \in \mathbf {V} \backslash (\{C \} \cup C h (C))} P (o | P a (O)) \delta_ {C = c ^ {-}}) \tag {4} \\ - P (e ^ {+} | c ^ {-}). \\ \end{array}
$$

$$
S E _ {\pi_ {1}} \left(c ^ {+}, c ^ {-}\right) \leq \tau , \quad S E _ {\pi_ {1}} \left(c ^ {-}, c ^ {+}\right) \leq \tau ,
$$

$$
\forall P a (E), \quad P ^ {\prime} (e ^ {-} | P a (E)) + P ^ {\prime} (e ^ {+} | P a (E)) = 1,
$$

$$
\forall P a (E), e, \quad P r ^ {\prime} (e | P a (E)) \geq 0,
$$
