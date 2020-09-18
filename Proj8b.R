rm(list=ls())
Sys.setlocale("LC_ALL","English")
Sys.setenv(LANGUAGE="en")

getwd()
setwd("C:/RFoldr/Part8/Project_Part8")

#libraries
library(caret)
training <- read.csv("training.csv")
testing <- read.csv("testing.csv")



str(training, list.len=ncol(training))

for (i in 8:159){
  if (class(training[,i])=="character"){
    print(class(training[i]))
    training[,i] <- as.numeric(training[,i])
  }
}
for (i in 8:159){
  if (class(testing[,i])=="character"){
    print(class(testing[i]))
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
###na's have been eliminated

###create factor response variables
training$classe <- factor(training$classe)


##keeping numerical variables
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


#partir en dos el data set
set.seed(12345) # For reproducibile purpose
inTrain <- createDataPartition(trainingnum$classe, p = 0.70, list = FALSE)
validationset <- trainingnum[-inTrain, ]
trainingset <- trainingnum[inTrain, ]
rm(inTrain)


#pca
trainingset.pca <- prcomp(trainingset[,-c(53)], center=TRUE, scale=TRUE)
summary(trainingset.pca)
trainset.scores <- predict(trainingset.pca,newdata=trainingset[,-c(53)])
trainset.scores <- as.data.frame(trainset.scores[,1:12])
trainset.scores$classe <- trainingset$classe

valset.scores <- predict(trainingset.pca,newdata=validationset[,-c(53)])
valset.scores <- as.data.frame(valset.scores[,1:12])
valset.scores$classe <- validationset$classe

#model fitting
set.seed(12345)
nuFoldData <- 10
nuRepeats <- 3
train_control <- trainControl(method = "repeatedcv", number = nuFoldData, repeats = nuRepeats)
trainedModel <- train(classe ~ ., data = trainset.scores,
                      method = "rf", ntree = nuFoldData, trControl = train_control)

print(trainedModel$finalModel)
trainedModel
confusionMatrix(trainedModel, newdata = predict(trainedModel, 
                                                newdata = valset.scores))

#prediction
testset.scores <- predict(trainingset.pca,newdata=testingnum[,-c(53)])
testset.scores <- as.data.frame(testset.scores[,1:12])
testset.scores$predicted <- predict(trainedModel, newdata = testset.scores)
testset.scores$predicted