---
title: "Practical Machine Learning Project"
author: "Guarob"
date: "17/9/2020"
output:
  html_document: 
    keep_md: yes
---

## Summary
This project aims to build a predictive model to assign a performance category (A, B, C, D, E) to observations of personal activity. The data comes from Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013 at http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. This report describes how the predictive model was built, how cross validation was used, what our expectation of out-of-sample error was, and why we made the choices we did. Additionally, predictions on a test dataset were estimated.

## Data
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. For this dataset, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants was collected. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data was splitted into a training set and a test set.

## Getting the data
The data was downloaded to a local folder with the following code:


```r
library(caret)
urltrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
destfiletrain <- "C:/RFoldr/Part8/Machine-Learning-Project/training.csv"
destfiletest <- "C:/RFoldr/Part8/Machine-Learning-Project/testing.csv"
download.file(urltrain, destfiletrain)
download.file(urltest, destfiletest)

training <- read.csv("training.csv")
testing <- read.csv("testing.csv")
```

## Data processing
The first step consisted of analyzing and adjusting the structure of the data, cleaning the dataset and selecting variables with a small number of missing values and that could provide information to the predictive model. We decided to convert character variables to numeric variables, and to keep -as part of this initial stage- variables with up to 10% of missing values. 
After removing variables with a significant portion of missing values, we ended up with a dataset containing 60 variables, including the response variable now coded as a factor variable. We decided to eliminate character variables or variables that do not add predictive power.

```r
for (i in 8:159){
  if (class(training[,i])=="character"){
    training[,i] <- as.numeric(training[,i])
  }
}
for (i in 8:159){
  if (class(testing[,i])=="character"){
    testing[,i] <- as.numeric(testing[,i])
  }
}
misvalpct <- matrix(nrow=length(names(training)),ncol=2)
for (i in 1:length(names(training))){
  misvalpct[i,1] <- i
  misvalpct[i,2] <- sum(is.na(training[,i]))/length(training$classe)
}
misvals <- data.frame(misvalpct)
colnames(misvals)<-c("colnum","misspct")
elimcols <- misvals[misvals$misspct > 0.10,1]
training <- training[,-elimcols]
testing <- testing[,-elimcols]
misvalpct <- NULL
rm(misvals)
rm(elimcols)

training$classe <- factor(training$classe)

misvals <- c()
for (i in 1:length(names(training))){
  if (is.character(training[,i])){
    misvals <- rbind(misvals,i)
  }
}
trainingnum <- training[,-misvals]
testingnum <- testing[,-misvals]
rm(misvals)
trainingnum <- trainingnum[,-c(1:4)]
testingnum <- testingnum[,-c(1:4)]
```

## Model building
The first issue to notice is that the dependent variable in this dataset is categorical with five different levels (A, B, C, D, E). In spite of A meaning the only correct way to perform the activity, the rest of the levels are not aggregated into a single group as they provide information about what is being done wrong. This type of response variable limits the type of model to build, discarding linear models and non-linear models such as a logistic regression. We will use methods based on decision trees to build our predictive model, as they can handle categorical variables.

### Partitioning the training dataset
In order to conduct cross validation, we dedided to split the training dataset into a training set (70% of observations) and a validation set (30%). The training set will be used to estimate the predictive model and the validation set will be used to estimate out-of-sample error rates.

```r
set.seed(12345)
inTrain <- createDataPartition(trainingnum$classe, p = 0.70, list = FALSE)
validationset <- trainingnum[-inTrain, ]
trainingset <- trainingnum[inTrain, ]
rm(inTrain)
```

### Variable Selection
We decided to reduce the dimension of the dataset by conducting a principal components analysis and keeping only those components that contribute, jointly, with more than 80% of the variance.


```r
trainingset.pca <- prcomp(trainingset[,-c(53)], center=TRUE, scale=TRUE)
summary(trainingset.pca)
```

```
## Importance of components:
##                           PC1    PC2     PC3     PC4     PC5     PC6     PC7
## Standard deviation     2.8957 2.8463 2.16273 2.07991 1.91557 1.72865 1.49241
## Proportion of Variance 0.1613 0.1558 0.08995 0.08319 0.07057 0.05747 0.04283
## Cumulative Proportion  0.1613 0.3170 0.40700 0.49019 0.56076 0.61822 0.66105
##                            PC8     PC9    PC10    PC11    PC12    PC13    PC14
## Standard deviation     1.43553 1.31316 1.23065 1.17263 1.05920 0.99462 0.93763
## Proportion of Variance 0.03963 0.03316 0.02913 0.02644 0.02158 0.01902 0.01691
## Cumulative Proportion  0.70068 0.73385 0.76297 0.78941 0.81099 0.83001 0.84692
##                           PC15    PC16    PC17    PC18   PC19    PC20    PC21
## Standard deviation     0.90212 0.88662 0.82010 0.74894 0.7209 0.69211 0.64069
## Proportion of Variance 0.01565 0.01512 0.01293 0.01079 0.0100 0.00921 0.00789
## Cumulative Proportion  0.86257 0.87769 0.89062 0.90141 0.9114 0.92062 0.92851
##                           PC22    PC23   PC24    PC25    PC26    PC27    PC28
## Standard deviation     0.62640 0.60927 0.5770 0.54836 0.54125 0.50450 0.48544
## Proportion of Variance 0.00755 0.00714 0.0064 0.00578 0.00563 0.00489 0.00453
## Cumulative Proportion  0.93606 0.94319 0.9496 0.95538 0.96101 0.96591 0.97044
##                           PC29    PC30    PC31    PC32    PC33    PC34    PC35
## Standard deviation     0.45308 0.41599 0.38915 0.36002 0.34774 0.33532 0.30225
## Proportion of Variance 0.00395 0.00333 0.00291 0.00249 0.00233 0.00216 0.00176
## Cumulative Proportion  0.97439 0.97772 0.98063 0.98312 0.98545 0.98761 0.98936
##                           PC36    PC37    PC38    PC39    PC40   PC41    PC42
## Standard deviation     0.28226 0.25347 0.23808 0.23463 0.20066 0.1911 0.18544
## Proportion of Variance 0.00153 0.00124 0.00109 0.00106 0.00077 0.0007 0.00066
## Cumulative Proportion  0.99090 0.99213 0.99322 0.99428 0.99506 0.9958 0.99642
##                           PC43    PC44    PC45    PC46    PC47    PC48    PC49
## Standard deviation     0.17944 0.16682 0.16540 0.16420 0.14823 0.14159 0.11333
## Proportion of Variance 0.00062 0.00054 0.00053 0.00052 0.00042 0.00039 0.00025
## Cumulative Proportion  0.99704 0.99757 0.99810 0.99862 0.99904 0.99943 0.99967
##                           PC50    PC51    PC52
## Standard deviation     0.09450 0.07705 0.04625
## Proportion of Variance 0.00017 0.00011 0.00004
## Cumulative Proportion  0.99984 0.99996 1.00000
```

```r
trainset.scores <- predict(trainingset.pca,newdata=trainingset[,-c(53)])
trainset.scores <- as.data.frame(trainset.scores[,1:12])
trainset.scores$classe <- trainingset$classe

valset.scores <- predict(trainingset.pca,newdata=validationset[,-c(53)])
valset.scores <- as.data.frame(valset.scores[,1:12])
valset.scores$classe <- validationset$classe
```

### Model Fitting
The random forest model uses k-mode cross validation and is run 3 times on 10-fold data starting with 70% of the data for training and 30% for validating the model.

Below the accuracy will test the prediction accuracy derived from the training data against the validation data


```r
set.seed(12345)
nuFoldData <- 10
nuRepeats <- 3
train_control <- trainControl(method = "repeatedcv", number = nuFoldData, repeats = nuRepeats)
trainedModel <- train(classe ~ ., data = trainset.scores,
                      method = "rf", ntree = nuFoldData, trControl = train_control)

print(trainedModel$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = ..1, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 10
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 14.74%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3497  115  102   93   56  0.09474502
## B  187 2162  126   75   76  0.17669459
## C  149  116 1958  103   48  0.17523168
## D  108   74  140 1857   51  0.16726457
## E   66  139   96   83 2112  0.15384615
```

```r
trainedModel
```

```
## Random Forest 
## 
## 13737 samples
##    12 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 12363, 12363, 12363, 12364, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9203373  0.8992021
##    7    0.9190267  0.8975681
##   12    0.9161639  0.8939329
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
confusionMatrix(trainedModel, newdata = predict(trainedModel, 
                                                newdata = valset.scores))
```

```
## Cross-Validated (10 fold, repeated 3 times) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 27.0  0.8  0.5  0.4  0.2
##          B  0.5 17.4  0.5  0.2  0.5
##          C  0.5  0.7 15.9  0.9  0.3
##          D  0.4  0.2  0.3 14.7  0.3
##          E  0.2  0.2  0.2  0.2 17.1
##                             
##  Accuracy (average) : 0.9203
```

With these results we expect to obtain an out-of-sample error rate of 8%.

# Prediction on test dataset

```r
testset.scores <- predict(trainingset.pca,newdata=testingnum[,-c(53)])
testset.scores <- as.data.frame(testset.scores[,1:12])
testset.scores$predicted <- predict(trainedModel, newdata = testset.scores)
testset.scores$predicted
```

```
##  [1] A A A A A E D A A A A C B A E E A B B B
## Levels: A B C D E
```


