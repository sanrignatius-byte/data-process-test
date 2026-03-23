# Element: 1602.05352_figure_1

- **type**: figure
- **doc_id**: 1602.05352
- **label**: Figure 1
- **page_idx**: 0

## Caption

Figure 1. Movie-Lovers toy example. Top row: true rating matrix $Y$ , propensity matrix $P$ , observation indicator matrix $O$ . Bottom row: two rating prediction matrices $\hat { Y _ { 1 } }$ and $\hat { Y } _ { 2 }$ , and intervention indicator matrix $\hat { Y } _ { 3 }$ .

## Content

Figure 1. Movie-Lovers toy example. Top row: true rating matrix $Y$ , propensity matrix $P$ , observation indicator matrix $O$ . Bottom row: two rating prediction matrices $\hat { Y _ { 1 } }$ and $\hat { Y } _ { 2 }$ , and intervention indicator matrix $\hat { Y } _ { 3 }$ .

## Context Before

(无)

## Context After

set of users are “horror lovers” who rate all horror movies 5 and all romance movies 1. Similarly, there is a subset of “romance lovers” who rate just the opposite way. However, both groups rate dramas as 3. The binary matrix ${ \cal O } \in \{ 0 , 1 \} ^ { \bar { U } \times \bar { I } }$ in Figure 1 shows for which movies the users provided their rating to the system, $\left[ O _ { u , i } \right. = 1 ] \Leftrightarrow$ $[ Y _ { u , i }$ observed]. Our toy example shows a strong correlation between liking and rating a movie, and the matrix $P$ describes the marginal probabilities $P _ { u , i } = P ( O _ { u , i } = 1 )$ with which each rating is revealed. For this data, consider the following two evaluation tasks.

3.1. Task 1: Estimating Rating Prediction Accuracy

For the first task, we want to evaluate how well a predicted rating matrix $\hat { Y }$ reflects the true ratings in $Y$ . Standard evaluation measures like Mean Absolute Error (MAE) or Mean Squared Error (MSE) can be written as:

Consider a toy example adapted from Steck (2010) to illustrate the disastrous effect that selection bias can have on conventional evaluation using a test set of held-out ratings. Denote with $u ~ \in ~ \{ 1 , . . . , U \}$ the users and with $i \in \{ 1 , . . . , I \}$ the movies. Figure 1 shows the matrix of true ratings $\dot { Y } \in \mathfrak { R } ^ { U \times I }$ for our toy example, where a sub-

set of users are “horror lovers” who rate all horror movies 5 and all romance mov

## Referring Paragraphs

1. Propensity-based approaches have been widely used in causal inference from observational studies (Imbens & Rubin, 2015), as well as in complete-case analysis for missing data (Little & Rubin, 2002; Seaman & White, 2013) and in survey sampling (Thompson, 2012). However, their use in matrix completion is new to our knowledge. Weighting approaches are also widely used in domain adaptation and covariate shift, where data from one source is used to train for a different problem (e.g., Huang et al., 2

2. set of users are “horror lovers” who rate all horror movies 5 and all romance movies 1. Similarly, there is a subset of “romance lovers” who rate just the opposite way. However, both groups rate dramas as 3. The binary matrix ${ \cal O } \in \{ 0 , 1 \} ^ { \bar { U } \times \bar { I } }$ in Figure 1 shows for which movies the users provided their rating to the system, $\left[ O _ { u , i } \right. = 1 ] \Leftrightarrow$ $[ Y _ { u , i }$ observed]. Our toy example shows a strong correlation bet

3. Consider a toy example adapted from Steck (2010) to illustrate the disastrous effect that selection bias can have on conventional evaluation using a test set of held-out ratings. Denote with $u ~ \in ~ \{ 1 , . . . , U \}$ the users and with $i \in \{ 1 , . . . , I \}$ the movies. Figure 1 shows the matrix of true ratings $\dot { Y } \in \mathfrak { R } ^ { U \times I }$ for our toy example, where a sub-

4. set of users are “horror lovers” who rate all horror movies 5 and all romance movies 1. Similarly, there is a subset of “romance lovers” who rate just the opposite way. However, both groups rate dramas as 3. The binary matrix ${ \cal O } \in \{ 0 , 1 \} ^ { \bar { U } \times \bar { I } }$ in Figure 1 shows for which movies the users provided their rating to the system, $\left[ O _ { u , i } \right. = 1 ] \Leftrightarrow$ $[ Y _ { u , i }$ observed]. Our toy example shows a strong correlation bet

5. We call this the naive estimator, and its naivety leads to a gross misjudgment for the $\hat { Y } _ { 1 }$ and $\hat { Y } _ { 2 }$ given in Figure 1. Even though $\hat { Y _ { 1 } }$ is clearly better than $\hat { Y } _ { 2 }$ by any reasonable measure of performance, $\hat { R } _ { n a i v e } ( \hat { Y } )$ will reliably claim that $\hat { Y } _ { 2 }$ has better MAE than $\hat { Y } _ { 1 }$ . This error is due to selection bias, since 1-star ratings are under-represented in the

6. Instead of evaluating the accuracy of predicted ratings, we may want to more directly evaluate the quality of a particular recommendation. To this effect, let’s redefine $\hat { Y }$ to now encode recommendations as a binary matrix analogous to $O$ , where $[ \hat { Y } _ { u , i } = 1 ] \Leftrightarrow [ i$ is recommended to $u ]$ , limited to a budget of $k$ recommendations per user. An example is $\hat { Y } _ { 3 }$ in Figure 1. A reasonable way to measure the quality of a recommendation is 

7. Figure 1.

8. We call this the naive estimator, and its naivety leads to a gross misjudgment for the $\hat { Y } _ { 1 }$ and $\hat { Y } _ { 2 }$ given in Figure 1.

9. An example is $\hat { Y } _ { 3 }$ in Figure 1.

10. Consider a toy example adapted from Steck (2010) to illustrate the disastrous effect that selection bias can have on conventional evaluation using a test set of held-out ratings. Denote with $u ~ \in ~ \{ 1 , . . . , U \}$ the users and with $i \in \{ 1 , . . . , I \}$ the movies. Figure 1 shows the matrix of true ratings $\dot { Y } \in \mathfrak { R } ^ { U \times I }$ for our toy example, where a sub-
