# x : p-dim covariates 
# y : outcome
setwd("~/GitHub/trees-and-forests/R/Test")
source("../Trees/DecisionTree.R")
x <- iris[,-5]
y <- iris$Species
clf_m <- DecisionTree$new(criterion="gini",
                          max_depth = 4,
                          min_node_size = NULL,
                          alpha_regular = NULL,
                          mtry = NULL,
                          random_state = NULL)
clf_m$fit(x,y)
result <- clf_m$predict(x)
table(y,result)
