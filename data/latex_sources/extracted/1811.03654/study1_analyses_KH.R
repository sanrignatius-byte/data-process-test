#Fairness Study 1

library(lme4)
if (!require("lsmeans")) {install.packages("lsmeans"); require("lsmeans")}
if (!require("emmeans")) {install.packages("emmeans"); require("emmeans")}
library(reshape2)
if (!require("effects")) {install.packages("effects"); require("effects")}
if (!require("reshape")) {install.packages("reshape"); require("reshape")}
if (!require("lme4")) {install.packages("lme4"); require("lme4")}

# ------------------------------------------------------------
# read in data & data cleaning

dat1 <- read.csv("f1.csv", na.strings = "", stringsAsFactors = FALSE)
head(dat1)
str(dat1)
names(dat1)
dat1 <- dat1[-1, ]
dat1 <- droplevels(dat1)
str(dat1)
names(dat1)

#creating between-subject variable

dat1$condition7040 <- ifelse(is.na(dat1$X70.40_1), 0, 1)
table(dat1$condition7040)

dat1$condition9010 <- ifelse(is.na(dat1$X90.10_1), 0, 1)
table(dat1$condition9010)

dat1$condition5550 <- ifelse(is.na(dat1$X50.55_1), 0, 1)
table(dat1$condition5550)

dat1$condition10020 <- ifelse(is.na(dat1$X100.20_1), 0, 1)
table(dat1$condition10020)

dat1$condition <- ifelse(dat1$condition7040 == 1, "7040", NA)
dat1$condition <- ifelse(dat1$condition9010 == 1, "9010", dat1$condition)
dat1$condition <- ifelse(dat1$condition5550 == 1, "5550", dat1$condition)
dat1$condition <- ifelse(dat1$condition10020 == 1, "10020", dat1$condition)

dat1$condition <- as.factor(dat1$condition)
table(dat1$condition)
names(dat1)
str(dat1)

dat1_complete <- dat1[complete.cases(dat1[ , 41]), ] #want only the observations with complete cases for variable 42
str(dat1_complete)

dat1_complete$X70.40_1 <- as.numeric(dat1_complete$X70.40_1)
dat1_complete$X70.40_2 <- as.numeric(dat1_complete$X70.40_2)
dat1_complete$X70.40_3 <- as.numeric(dat1_complete$X70.40_3)
dat1_complete$X100.20_1  <- as.numeric(dat1_complete$X100.20_1)
dat1_complete$X100.20_2  <- as.numeric(dat1_complete$X100.20_2)
dat1_complete$X100.20_3  <- as.numeric(dat1_complete$X100.20_3)
dat1_complete$X50.55_1  <- as.numeric(dat1_complete$X50.55_1)
dat1_complete$X50.55_2  <- as.numeric(dat1_complete$X50.55_2)
dat1_complete$X50.55_3  <- as.numeric(dat1_complete$X50.55_3)
dat1_complete$X90.10_1 <- as.numeric(dat1_complete$X90.10_1)
dat1_complete$X90.10_2 <- as.numeric(dat1_complete$X90.10_2)
dat1_complete$X90.10_3 <- as.numeric(dat1_complete$X90.10_3)

str(dat1_complete)

#creating within-subject variable

dat1_complete$allA <- ifelse(dat1_complete$condition == "7040", dat1$X70.40_1, NA)
dat1_complete$allA <- ifelse(dat1_complete$condition == "5550", dat1_complete$X50.55_1, dat1_complete$allA)
dat1_complete$allA <- ifelse(dat1_complete$condition == "10020", dat1_complete$X100.20_1, dat1_complete$allA)
dat1_complete$allA <- ifelse(dat1_complete$condition == "9010", dat1_complete$X90.10_1, dat1_complete$allA)

print(dat1_complete$allA)
dat1_complete$allA <- as.numeric(dat1_complete$allA)

dat1_complete$split5050 <- ifelse(dat1_complete$condition == "7040", dat1_complete$X70.40_2, NA)
dat1_complete$split5050 <- ifelse(dat1_complete$condition == "5550", dat1_complete$X50.55_2, dat1_complete$split5050)
dat1_complete$split5050 <- ifelse(dat1_complete$condition == "10020", dat1_complete$X100.20_2, dat1_complete$split5050)
dat1_complete$split5050 <- ifelse(dat1_complete$condition == "9010", dat1_complete$X90.10_2, dat1_complete$split5050)

print(dat1_complete$split5050)

dat1_complete$ratio <- ifelse(dat1_complete$condition == "7040", dat1_complete$X70.40_3, NA)
dat1_complete$ratio <- ifelse(dat1_complete$condition == "5550", dat1_complete$X50.55_3, dat1_complete$ratio)
dat1_complete$ratio <- ifelse(dat1_complete$condition == "10020", dat1_complete$X100.20_3, dat1_complete$ratio)
dat1_complete$ratio <- ifelse(dat1_complete$condition == "9010", dat1_complete$X90.10_3, dat1_complete$ratio)

print(dat1_complete$ratio)

str(dat1_complete)
names(dat1_complete)

# id.vars = not reshape
# measure.vars = reshape
# need to convert to long format, so 3 observations per subject

dat1_long <- melt(dat1_complete, measure.vars = c("allA", "split5050", "ratio"), variable.name = "within", value.name = "response")

str(dat1_long)
head(dat1_long)

# ------------------------------------------------------------
# estimation

model1 <- lmer(response ~ within * condition + (1 | ResponseId), data = dat1_long)

# ------------------------------------------------------------
# post-estimation

#12 cells
#3(within-subject) x 4(between-subject) design

summary(model1)

MeansWithin <- lsmeans(model1, specs = ~ within | condition)
MeansWithin
contrast(MeansWithin, method = "pairwise", asjust = "holm") #sequential bonferroni adjustment
#pairwise comparisons across within-subject conditions, within the other variable, using a sequential bonferroni adjustment

#55% and 50% (Treatment 1)
#70% and 40% (Treatment 2)
#90% and 10% (Treatment 3)
#100% and 20% (Treatment 4)
