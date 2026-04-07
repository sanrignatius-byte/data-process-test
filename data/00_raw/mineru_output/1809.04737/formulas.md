$$
\mathbb {L} (f) = \mathbb {E} _ {\mathbf {X}, Y} [ \mathbb {1} _ {f (\mathbf {x}) \neq y} ],
$$

$$
\min  _ {f \in \mathcal {F}} \mathbb {L} (f) = \min  _ {f \in \mathcal {F}} \mathbb {E} _ {\mathbf {X}, Y} [ \mathbb {1} _ {f (\mathbf {x}) \neq y} ].
$$

$$
\begin{array}{l} \mathbb {L} (f) = \mathbb {E} _ {\mathbf {X}, Y} \left[ \mathbb {1} _ {\operatorname {s i g n} (h (\mathbf {x})) \neq y} \right] \\ = \mathbb {E} _ {\mathbf {X}} \big [ P r (Y = 1 | \mathbf {x}) \mathbb {1} _ {h (\mathbf {x}) <   0} + P r (Y = - 1 | \mathbf {x}) \mathbb {1} _ {h (\mathbf {x}) > 0} \big ]. \\ \end{array}
$$

$$
\begin{array}{l} \mathbb {L} _ {\phi} (h) = \mathbb {E} _ {\mathbf {X}} \left[ P r (Y = 1 | \mathbf {x}) \phi (h (\mathbf {x})) \right. \\ \left. + \big (1 - P r (Y = 1 | \mathbf {x}) \big) \phi \big (- h (\mathbf {x}) \big) \right], \\ \end{array}
$$

$$
\min  _ {h \in \mathcal {H}} \mathbb {L} _ {\phi} (h). \tag {1}
$$

$$
\mathbb {R D} (f) = \mathbb {E} _ {\mathbf {X} | S = s ^ {+}} [ \mathbb {1} _ {f (\mathbf {x}) = 1} ] - \mathbb {E} _ {\mathbf {X} | S = s ^ {-}} [ \mathbb {1} _ {f (\mathbf {x}) = 1} ].
$$

$$
\min  _ {h \in \mathcal {H}} \quad \mathbb {L} _ {\phi} (h) \tag {2}
$$

$$
\begin{array}{l} \mathbb {R} \mathbb {D} (f) = \mathbb {E} _ {\mathbf {X} | S = s ^ {+}} \left[ \mathbb {1} \left[ \operatorname {s i g n} (h (\mathbf {x})) = 1 \right] \right] \\ - \mathbb {E} _ {\mathbf {X} | S = s ^ {-}} \left[ \mathbb {1} \left[ \operatorname {s i g n} (h (\mathbf {x})) = 1 \right] \right] \\ = \mathbb {E} _ {\mathbf {X} | S = s ^ {+}} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ] - \mathbb {E} _ {\mathbf {X} | S = s ^ {-}} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ] \\ = \mathbb {E} _ {\mathbf {X} | S = s +} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ] + \mathbb {E} _ {\mathbf {X} | S = s -} [ \mathbb {1} _ {h (\mathbf {x}) <   0} ] - 1. \\ \end{array}
$$

$$
\begin{array}{l} \mathbb {R D} (f) = \mathbb {E} _ {\mathbf {X}} \left[ \frac {P (S = s ^ {+} | \mathbf {x})}{P (S = s ^ {+})} \mathbb {1} _ {h (\mathbf {x}) > 0} \right. \tag {3} \\ + \frac {P (S = s ^ {-} | \mathbf {x})}{P (S = s ^ {-})} \mathbb {1} _ {h (\mathbf {x}) <   0} - 1 ]. \\ \end{array}
$$

$$
\min  _ {h \in \mathcal {H}} \quad \mathbb {L} _ {\phi} (h)
$$

$$
\begin{array}{l} \mathbb {L} _ {\phi} (h) = \mathbb {E} _ {\mathbf {X}} \left[ P r (Y = 1 | \mathbf {x}) \phi (h (\mathbf {x})) \right. \\ \left. + \left(1 - P r (Y = 1 | \mathbf {x})\right) \phi \big (- h (\mathbf {x}) \big) \right], \\ \end{array}
$$

$$
\mathbb {R D} _ {\kappa} (h) = \mathbb {E} _ {\mathbf {X}} \big [ \frac {\eta (\mathbf {x})}{p} \kappa \big (h (\mathbf {x}) \big) + \frac {1 - \eta (\mathbf {x})}{1 - p} \kappa \big (- h (\mathbf {x}) \big) - 1 \big ],
$$

$$
\mathbb {R D} _ {\delta} (h) = \mathbb {E} _ {\mathbf {X}} \big [ \frac {\eta (\mathbf {x})}{p} \delta \big (h (\mathbf {x}) \big) + \frac {1 - \eta (\mathbf {x})}{1 - p} \delta \big (- h (\mathbf {x}) \big) - 1 \big ].
$$

$$
\begin{array}{l} f _ {\max } (\mathbf {x}) = \left\{ \begin{array}{l l} 1 & \text {i f} \eta (\mathbf {x}) \geq p, \\ - 1 & \text {o t h e r w i s e}, \end{array} \right. \\ f _ {\min } (\mathbf {x}) = \left\{ \begin{array}{c l} - 1 & i f \eta (\mathbf {x}) \geq p, \\ 1 & o t h e r w i s e. \end{array} \right. \\ \end{array}
$$

$$
C ^ {\eta} \big (h (\mathbf {x}) \big) = \frac {\eta (\mathbf {x})}{p} \mathbb {1} _ {h (\mathbf {x}) > 0} + \frac {1 - \eta (\mathbf {x})}{1 - p} \mathbb {1} _ {h (\mathbf {x}) <   0} - 1.
$$

$$
C _ {\kappa} ^ {\eta} \big (h (\mathbf {x}) \big) = \frac {\eta (\mathbf {x})}{p} \kappa \big (h (\mathbf {x}) \big) + \frac {1 - \eta (\mathbf {x})}{1 - p} \kappa \big (- h (\mathbf {x}) \big) - 1,
$$

$$
C ^ {\eta} (\alpha) = \frac {\eta}{p} \mathbb {1} _ {\alpha > 0} + \frac {1 - \eta}{1 - p} \mathbb {1} _ {\alpha <   0} - 1,
$$

$$
C _ {\kappa} ^ {\eta} (\alpha) = \frac {\eta}{p} \kappa (\alpha) + \frac {1 - \eta}{1 - p} \kappa (- \alpha) - 1,
$$

$$
H ^ {-} (\eta) = \min  _ {\alpha \in R} C ^ {\eta} (\alpha) = \min  _ {\alpha \in R} \left[ \frac {\eta}{p} \mathbb {1} _ {\alpha > 0} + \frac {1 - \eta}{1 - p} \mathbb {1} _ {\alpha <   0} - 1 \right],
$$

$$
H _ {\kappa} ^ {-} (\eta) = \min  _ {\alpha \in R} C _ {\kappa} ^ {\eta} (\alpha) = \min  _ {\alpha \in R} \left[ \frac {\eta}{p} \kappa (\alpha) + \frac {1 - \eta}{1 - p} \kappa (- \alpha) - 1 \right]. \tag {4}
$$

$$
\mathbb {R D} _ {\kappa} ^ {-} = \mathbb {E} _ {\mathbf {X}} \left[ H _ {\kappa} ^ {-} (\eta (\mathbf {x})) \right].
$$

$$
H _ {\kappa} ^ {\circ} (\eta) = \min  _ {\alpha : \alpha (\eta - p) \geq 0} C _ {\kappa} ^ {\eta} (\alpha). \tag {5}
$$

$$
\psi_ {\kappa} \left(\mathbb {R D} (h) - \mathbb {R D} ^ {-}\right) \leq \mathbb {R D} _ {\kappa} (h) - \mathbb {R D} _ {\kappa} ^ {-},
$$

$$
\psi_ {\delta} \left(\mathbb {R D} ^ {+} - \mathbb {R D} (h)\right) \leq \mathbb {R D} _ {\kappa} ^ {+} - \mathbb {R D} _ {\kappa} (h),
$$

$$
\psi_ {\kappa} (\mu) = H _ {\kappa} ^ {\circ} \big (p (1 - p) \mu + p \big) - H _ {\kappa} ^ {-} \big (p (1 - p) \mu + p \big),
$$

$$
\psi_ {\delta} (\mu) = H _ {\delta} ^ {+} (p (1 - p) \mu + p) - H _ {\delta} ^ {\circ} (p (1 - p) \mu + p).
$$

$$
\mathbb {R D} (f) \leq \mathbb {R D} ^ {-} + \psi_ {\kappa} ^ {- 1} \left(\mathbb {R D} _ {\kappa} (h) - \mathbb {R D} _ {\kappa} ^ {-}\right),
$$

$$
\mathbb {R D} (f) \geq \mathbb {R D} ^ {+} - \psi_ {\delta} ^ {- 1} \big (\mathbb {R D} _ {\delta} ^ {+} - \mathbb {R D} _ {\delta} (h) \big).
$$

$$
\min  _ {h \in \mathcal {H}} \quad \mathbb {L} _ {\phi} (h) \tag {6}
$$

$$
- \mathbb {R} \mathbb {D} _ {\delta} (h) \leq \psi_ {\delta} \left(c _ {2} + \mathbb {R} \mathbb {D} ^ {+}\right) + \mathbb {R} \mathbb {D} _ {\delta} ^ {+}.
$$

$$
\mathbb {R D} (f _ {\max }) = \mathbb {E} _ {\mathbf {X}} \left[ \frac {\eta (\mathbf {x})}{p} \mathbb {1} _ {f _ {\max } (\mathbf {X}) = 1} + \frac {1 - \eta (\mathbf {x})}{1 - p} \mathbb {1} _ {f _ {\max } (\mathbf {X}) = - 1} \right] - 1.
$$

$$
\begin{array}{l} \mathbb {R D} (f _ {\max }) - \mathbb {R D} (f) = \mathbb {E} _ {\mathbf {X}} \left[ \frac {\eta (\mathbf {x})}{p} \left[ \mathbb {1} _ {f _ {\max } (\mathbf {X}) = 1} - \mathbb {1} _ {f (\mathbf {x}) = 1} \right] \right. \\ + \frac {1 - \eta (\mathbf {x})}{1 - p} \left[ \mathbb {1} _ {f _ {\max } (\mathbf {X}) = - 1} - \mathbb {1} _ {f (\mathbf {x}) = - 1} \right]. \\ \end{array}
$$

$$
\begin{array}{l} D C (\mathbf {x}) = \frac {\eta (\mathbf {x})}{p} \left[ \mathbb {1} _ {f _ {\max } (\mathbf {X}) = 1} - \mathbb {1} _ {f (\mathbf {x}) = 1} \right] \\ + \frac {1 - \eta (\mathbf {x})}{1 - p} \left[ \mathbb {1} _ {f _ {\max } (\mathbf {X}) = - 1} - \mathbb {1} _ {f (\mathbf {x}) = - 1} \right], \\ \end{array}
$$

$$
\begin{array}{l} H _ {\kappa} ^ {\circ} (\eta) = \min  _ {\alpha : \alpha (\eta - p) \geq 0} \frac {\eta}{p} \kappa (\alpha) + \frac {1 - \eta}{1 - p} \kappa (- \alpha) \\ = \min  _ {\alpha : \alpha (\eta - p) \geq 0} \left(\frac {\eta}{p} + \frac {1 - \eta}{1 - p}\right) \\ \left[ \frac {\frac {\eta}{p}}{\frac {\eta}{p} + \frac {1 - \eta}{1 - p}} \kappa (\alpha) + \frac {\frac {1 - \eta}{1 - p}}{\frac {\eta}{p} + \frac {1 - \eta}{1 - p}} \kappa (- \alpha) \right]. \\ \end{array}
$$

$$
H _ {\kappa} ^ {\circ} (\eta) = \min  _ {\alpha : \alpha (\eta - p) \geq 0} \nu \times \left[ \frac {\eta}{p \nu} \kappa (\alpha) + \frac {1 - \eta}{(1 - p) \nu} \kappa (- \alpha) \right].
$$

$$
\begin{array}{l} H _ {\phi} ^ {\circ} (\eta) \geq \min  _ {\alpha : \alpha (\eta - p) \geq 0} \nu \times \kappa \left(\frac {\eta}{p \nu} \alpha - \frac {1 - \eta}{(1 - p) \nu} \alpha\right) \\ = \min _ {\alpha : \alpha (\eta - p) \geq 0} \nu \times \kappa \left(\frac {\alpha (\eta - p)}{\nu * p (1 - p)}\right) \geq \nu \kappa (0). \\ \end{array}
$$

$$
H _ {\kappa} ^ {\circ} (\eta) = \left(\frac {\eta}{p} + \frac {1 - \eta}{1 - p}\right) \kappa (0).
$$

$$
H _ {\kappa} ^ {\circ} (p (1 - p) \mu + p) = (\mu - 2 p \mu + 2) \kappa (0).
$$

$$
\begin{array}{l} \mathbb {R D} (h) - \mathbb {R D} ^ {-} = \mathbb {E} _ {\mathbf {x}} \left[ C ^ {\eta} (h, \mathbf {x}) \right] - \min  _ {h \in \mathcal {H}} \mathbb {E} _ {\mathbf {x}} \left[ C ^ {\eta} (h, \mathbf {x}) \right] \\ = \mathbb {E} _ {\mathbf {x}} \left[ C ^ {\eta} (h, \mathbf {x}) - \min  _ {h \in \mathcal {H}} C ^ {\eta} (h, \mathbf {x}) \right] \\ = \mathbb {E} _ {\mathbf {x}} \left[ \mathbb {1} _ {(\eta - p) h (\mathbf {x}) <   0} \times \left[ \frac {| \eta - p |}{p (1 - p)} \right] \right] = \mathbb {E} _ {\mathbf {x}} \left[ g (\mathbf {x}) \right]. \\ \end{array}
$$

$$
\begin{array}{l} \psi_ {\kappa} \left(\mathbb {R} \mathbb {D} (h) - R D ^ {-}\right) = \psi_ {\kappa} \left(\mathbb {E} _ {\mathbf {x}} [ g (\mathbf {x}) ]\right) \leq \mathbb {E} _ {\mathbf {x}} \left[ \psi_ {\kappa} (g (\mathbf {x})) \right] \\ \leq \mathbb {E} _ {\mathbf {x}} \left[ \psi_ {\kappa} \left(\mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \left[ \frac {| \eta - p |}{p (1 - p)} \right]\right) \right] \\ = \mathbb {E} _ {\mathbf {x}} \left[ \mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \times \psi_ {\kappa} \left(\frac {| \eta - p |}{p (1 - p)}\right) \right] \\ = \mathbb {E} _ {\mathbf {x}} \left[ \mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \times \left[ H _ {\kappa} ^ {\circ} (\eta) - H _ {\kappa} ^ {-} (\eta) \right] \right] \\ = \mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \times \mathbb {E} _ {\mathbf {x}} \big [ H _ {\kappa} ^ {\circ} (\eta) - H _ {\kappa} ^ {-} (\eta) \big ]. \\ \end{array}
$$

$$
\begin{array}{l} \psi_ {\kappa} \left(\mathbb {R D} (h) - \mathbb {R D} ^ {-}\right) \leq \mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \\ \times \mathbb {E} _ {\mathbf {x}} \left[ H _ {\kappa} ^ {\circ} (\eta) - H _ {\kappa} ^ {-} (\eta) \right] + \mathbb {1} _ {(\eta - p) h (\mathbf {x}) \leq 0} \times 0 \\ \leq \mathbb {1} _ {(\eta - p) h (\mathbf {x}) > 0} \times \mathbb {E} _ {\mathbf {x}} \left[ C _ {\kappa} ^ {\eta} (h (\mathbf {x})) - H _ {\kappa} ^ {-} (\eta) \right] \\ + \mathbb {1} _ {(\eta - p) h (\mathbf {x}) \leq 0} \times \mathbb {E} _ {\mathbf {x}} \left[ C _ {\kappa} ^ {\eta} (h (\mathbf {x})) - H _ {\kappa} ^ {-} (\eta) \right] \\ = \mathbb {E} _ {\mathbf {x}} \left[ C _ {\kappa} ^ {\eta} (h (\mathbf {x})) - H _ {\kappa} ^ {-} (\eta) \right] = \mathbb {R D} _ {\kappa} (h) - \mathbb {R D} _ {\kappa} ^ {-}. \\ \end{array}
$$

$$
\mathbb {R R} (h) = \frac {\mathbb {E} _ {\mathbf {X} | S = s ^ {+}} \left[ \mathbb {1} _ {h (\mathbf {x}) > 0} \right]}{\mathbb {E} _ {\mathbf {X} | S = s ^ {-}} \left[ \mathbb {1} _ {h (\mathbf {x}) > 0} \right]}.
$$

$$
\mathbb {R R} (h) = \frac {\mathbb {E} _ {\mathbf {X} | S = s ^ {+}} \left[ \mathbb {1} _ {h (\mathbf {x}) > 0} \right]}{\mathbb {E} _ {\mathbf {X} | S = s ^ {-}} \left[ \mathbb {1} _ {h (\mathbf {x}) > 0} \right]} \leq \tau .
$$

$$
\mathbb {E} _ {\mathbf {X}} \left[ \frac {\eta}{p} \mathbb {1} _ {h (\mathbf {x}) > 0} + \tau \frac {1 - \eta (\mathbf {x})}{1 - p} \mathbb {1} _ {h (\mathbf {x}) > 0} \right] - \tau \leq 0. \tag {7}
$$

$$
\mathbb {E} \mathbb {O} (h) = \mathbb {E} _ {\mathbf {X} | S = s ^ {+}, Y} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ] - \mathbb {E} _ {\mathbf {X} | S = s ^ {-}, Y} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ].
$$

$$
\begin{array}{l} \mathbb {E} \mathbb {O} (h) = \mathbb {E} _ {\mathbf {X} | S = s ^ {+}, Y} [ \mathbb {1} _ {h (\mathbf {x}) > 0} ] + \mathbb {E} _ {\mathbf {X} | S = s ^ {-}, Y} [ \mathbb {1} _ {h (\mathbf {x}) <   0} ] - 1 \\ = \mathbb {E} _ {\mathbf {X} | Y} \left[ \frac {P (S = s ^ {+} | \mathbf {x} , y)}{P (S = s ^ {+} | y)} \mathbb {1} _ {h (\mathbf {x}) > 0} \right. \\ \left. + \frac {1 - P (S = s ^ {+} | \mathbf {x} , y)}{1 - P (S = s ^ {+} | y)} \mathbb {1} _ {h (\mathbf {x}) <   0} \right] - 1 \leq \tau . \tag {8} \\ \end{array}
$$

$$
\begin{array}{l} \mathbb {E} \mathbb {O P} (h) = \mathbb {E} _ {\mathbf {X} | Y = 1} \left[ \frac {P (S = s ^ {+} | \mathbf {x} , Y = 1)}{P (S = s ^ {+} | Y = 1)} \mathbb {1} _ {h (\mathbf {x}) > 0} \right. \\ \left. + \frac {1 - P (S = s ^ {+} | \mathbf {x} , Y = 1)}{1 - P (S = s ^ {+} | Y = 1)} \mathbb {1} _ {h (\mathbf {x}) <   0} \right] - 1 \leq \tau . \tag {9} \\ \end{array}
$$
