# Element: 1602.05352_formula_1

- **type**: formula
- **doc_id**: 1602.05352
- **label**: Formula
- **page_idx**: 0

## Caption

(无)

## Content

$$\begin{array}{l} \mathbb {E} _ {O} \Big [ \hat {R} _ {I P S} (\hat {Y} | P) \Big ] = \frac {1}{U \cdot I} \sum_ {u} \sum_ {i} \mathbb {E} _ {O _ {u, i}} \bigg [ \frac {\delta_ {u , i} (Y , \hat {Y})}{P _ {u , i}} O _ {u, i} \bigg ] \\ = \frac {1}{U \cdot I} \sum_ {u} \sum_ {i} \delta_ {u, i} (Y, \hat {Y}) = R (\hat {Y}). \\ \end{array}$$

## Context Before

(无)

## Context After

Consider a toy example adapted from Steck (2010) to illustrate the disastrous effect that selection bias can have on conventional evaluation using a test set of held-out ratings. Denote with $u ~ \in ~ \{ 1 , . . . , U \}$ the users and with $i \in \{ 1 , . . . , I \}$ the movies. Figure 1 shows the matrix of true ratings $\dot { Y } \in \mathfrak { R } ^ { U \times I }$ for our toy example, where a sub-

set of users are “horror lovers” who rate all horror movies 5 and all romance movies 1. Similarly, there is a subset of “romance lovers” who rate just the opposite way. However, both groups rate dramas as 3. The binary matrix ${ \cal O } \in \{ 0 , 1 \} ^ { \bar { U } \times \bar { I } }$ in Figure 1 shows for which movies the users provided their rating to the system, $\left[ O _ { u , i } \right. = 1 ] \Leftrightarrow$ $[ Y _ { u , i }$ observed]. Our toy example shows a strong correlation bet

We call this the naive estimator, and its naivety leads to a gross misjudgment for the $\hat { Y } _ { 1 }$ and $\hat { Y } _ { 2 }$ given in Figure 1. Even though $\hat { Y _ { 1 } }$ is clearly better than $\hat { Y } _ { 2 }$ by any reasonable measure of performance, $\hat { R } _ { n a i v e } ( \hat { Y } )$ will reliably claim that $\hat { Y } _ { 2 }$ has better MAE than $\hat { Y } _ { 1 }$ . This error is due to selection bias, since 1-star ratings are under-represented in the
