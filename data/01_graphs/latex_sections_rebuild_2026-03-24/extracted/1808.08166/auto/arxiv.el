(TeX-add-style-hook
 "arxiv"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "9pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("algorithmic" "noend")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "intro"
    "defs"
    "empirical"
    "conclusions"
    "appendix"
    "article"
    "art10"
    "booktabs"
    "subfigure"
    "algorithm"
    "algorithmic"
    "fullpage"
    "amsmath"
    "amssymb"
    "amsthm"
    "mathtools"
    "color"
    "xcolor"
    "natbib"
    "authblk"
    "hyperref"
    "xspace"
    "cleveref")
   (TeX-add-symbols
    '("INDSTATE" ["argument"] 0)
    '("Prob" 2)
    '("prob" 1)
    '("Ex" 2)
    '("ex" 1)
    '("mk" 1)
    '("ar" 1)
    '("sn" 1)
    '("sw" 1)
    "ZZ"
    "NN"
    "RR"
    "cA"
    "cB"
    "cE"
    "cF"
    "cL"
    "cG"
    "cH"
    "cK"
    "cM"
    "cP"
    "cQ"
    "cR"
    "cX"
    "cD"
    "cU"
    "cN"
    "cS"
    "cV"
    "cI"
    "knrw"
    "MSR"
    "poly"
    "polylog"
    "Expectation"
    "Probability"
    "FFP"
    "FP"
    "SP"
    "PA"
    "PY"
    "PYD"
    "PDO"
    "PXD"
    "VCD"
    "Reg"
    "LC"
    "CSC"
    "spsize"
    "spdisp"
    "fpsize"
    "fpdisp"
    "eps"
    "OPT"
    "argmin"
    "argmax"
    "epsilon")
   (LaTeX-add-bibliographies
    "./arxiv.bbl")
   (LaTeX-add-amsthm-newtheorems
    "theorem"
    "lemma"
    "fact"
    "claim"
    "remark"
    "corollary"
    "prop"
    "assumpt"
    "definition")
   (LaTeX-add-color-definecolors
    "DarkGreen"
    "DarkRed"
    "DarkBlue"))
 :latex)

