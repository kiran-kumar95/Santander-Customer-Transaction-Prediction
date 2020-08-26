rm(list=ls(all=T))

#Load Libraries
library(tidyverse) 
library(data.table)
library(glmnet)
library(ggplot2)
library(caret)
library(lightgbm)

# Reading the csv files
setwd("C:/Users/kiran/Desktop/python practice/Santander")
train = read.csv("train.csv", header = T)
test=read.csv("test.csv", header = T)
head(train)

str(train)
colnames(train)

# Missing value analysis
colSums(is.na(train))


#Target variable visualisation
table(train$target)

target_df <- data.frame(table(train$target))
colnames(target_df) <- c("target", "freq")
ggplot(data=target_df, aes(x=target, y=freq, fill=target)) +
  geom_bar(position = 'dodge', stat='identity', alpha=0.5) +
  scale_fill_manual("legend", values = c("1" = "dodgerblue", "0"="firebrick1")) +
  theme_classic()


#Correlations
cormat <- cor(train[,-c(1,2)])
summary(cormat[upper.tri(cormat)]) #Correlations between features nearly zero.

#Modeling
#--------------------Training Data----------------------------------

#Logistic Regression
X_train <- scale(train[,-(1:2)]) %>% data.frame
X_test <- scale(test[,-1]) %>% data.frame
target <- train$target

fit.logit <- glm(target~., data=X_train, family=binomial)
pred.logit <- predict(fit.logit, newdata=X_test, type="response")
submission <- read.csv("test.csv")
submission$target <- pred.logit
write.csv(submission, file="submission_logit.csv", row.names=F)


#light gbm
trainY <- train[,target]
trainX <- train[, !c("target", "ID_code"), with = F]
testX <- test[, !c("ID_code"), with = F]

# Create LGB Dataset
lgb.train <- lgb.Dataset(data=as.matrix(trainX), label=trainY)

lgb.grid <- list(objective = "binary",
                 metric = "auc",
                 min_sum_hessian_in_leaf = 1, # min_child_weight
                 feature_fraction = 0.1, # colsample_bytree
                 bagging_fraction = 0.5, # subsample
                 bagging_freq = 1, # stochastic nature of the training
                 min_data_in_leaf = 10,
                 #max_bin = 50,
                 #lambda_l1 = 8,
                 lambda_l2 = 1,
                 #min_data_in_bin=100,
                 min_gain_to_split = 10, # gamma
                 is_unbalance = TRUE # scale_pos_weight can be used instead
)

set.seed(777)
lgb.model.cv = lgb.cv(params = lgb.grid,
                      data = lgb.train,
                      learning_rate = 0.02,
                      num_leaves = 25,
                      max_depth = 2,
                      nrounds = 50000,
                      early_stopping_rounds = 1000,
                      eval_freq = 100,
                      eval = "auc",
                      nfold = 10,
                      stratified = TRUE)

best.iter <- lgb.model.cv$best_iter
cat(paste("Best num of iterations:"), best.iter)
cat(paste("Best score:", round(lgb.model.cv$best_score, 4))) # Best num of iterations: 10362, Best score: 0.8995

# train model with best iter
lgb.model = lgb.train(params = lgb.grid, 
                      data = lgb.train, 
                      learning_rate = 0.02,
                      num_leaves = 25,
                      max_depth = 2,
                      nrounds = best.iter,
                      eval_freq = 10, 
                      eval = "auc")

# predictions and submission
pred_sub <- predict(lgb.model, as.matrix(testX))
submission <- read.csv("test.csv")
submission$target <- pred_sub
write.csv(submission, file="submission_LGBM.csv", row.names=F)

