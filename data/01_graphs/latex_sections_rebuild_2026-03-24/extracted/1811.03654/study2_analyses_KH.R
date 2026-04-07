#Fairness Study 2

library(lme4)
if (!require("lsmeans")) {install.packages("lsmeans"); require("lsmeans")}
if (!require("emmeans")) {install.packages("emmeans"); require("emmeans")}
library(reshape2)
if (!require("effects")) {install.packages("effects"); require("effects")}

dat2 <- read.csv("f2_data.csv")
head(dat2)
str(dat2)
dat2 <- dat2[c(-1, -2), ]
dat2 <- droplevels(dat2)
str(dat2)

str(dat2)
dat2$condition <- as.factor(dat2$condition)
table(dat2$condition)

dat2$allA <- ifelse(dat2$condition == "7040", dat2$X7040_1, NA)
dat2$allA <- ifelse(dat2$condition == "5550", dat2$X5550_1, dat2$allA)
dat2$allA <- ifelse(dat2$condition == "10020", dat2$X10020_1, dat2$allA)
dat2$allA <- ifelse(dat2$condition == "9010", dat2$X9010_1, dat2$allA)

head(dat2$allA)
table(dat2$allA)

dat2$split5050 <- ifelse(dat2$condition == "7040", dat2$X7040_2, NA)
dat2$split5050 <- ifelse(dat2$condition == "5550", dat2$X5550_2, dat2$split5050)
dat2$split5050 <- ifelse(dat2$condition == "10020", dat2$X10020_2, dat2$split5050)
dat2$split5050 <- ifelse(dat2$condition == "9010", dat2$X9010_2, dat2$split5050)

table(dat2$split5050)

dat2$ratio <- ifelse(dat2$condition == "7040", dat2$X7040_3, NA)
dat2$ratio <- ifelse(dat2$condition == "5550", dat2$X5550_3, dat2$ratio)
dat2$ratio <- ifelse(dat2$condition == "10020", dat2$X10020_3, dat2$ratio)
dat2$ratio <- ifelse(dat2$condition == "9010", dat2$X9010_3, dat2$ratio)

table(dat2$ratio)

str(dat2)

dat2_long <- melt(dat2, measure.vars = c("allA", "split5050", "ratio"), variable.name = "within", value.name = "response")

str(dat2_long)

#need a factor that represents the race treatment variable
#cannot tell from data

# ------------------------------------------------------------
# estimation

model2 <- lmer(response ~ within * condition + (1 | ResponseId), data = dat2_long)

# ------------------------------------------------------------
# post-estimation

#24 cells
#3(within-subject) 
# x 4(between-subject factor similarity metric) design x 2(between-subject factor race) design

summary(model2)

#compare within subjects
MeansWithin_2 <- lsmeans(model2, specs = ~ within | condition)
MeansWithin_2
contrast(MeansWithin_2, method = "pairwise", asjust = "holm") #sequential bonferroni adjustment
#pairwise comparisons across within-subject conditions, within the other variable, using a sequential bonferroni adjustment

#55% and 50% (Treatment 1)
#70% and 40% (Treatment 2)
#90% and 10% (Treatment 3)
#100% and 20% (Treatment 4)
