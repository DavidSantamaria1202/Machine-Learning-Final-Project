library(caret)

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","TrainingData.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","TestData.csv")
setwd("C:/Users/Santa/Desktop/machine learning")

training <- read.table(file = "TrainingData.csv",header = TRUE,sep = ",")
testing <- read.table(file = "TestData.csv",header = TRUE,sep = ",")

training$classe <- as.factor(training$classe)

## Clean NA variables

Nas <- data.frame(variables = names(training),isNA = mapply(FUN=function(x){sum(is.na(x))},training,SIMPLIFY = TRUE))
NewDF <- training[,which(Nas$isNA == 0)]

testingDF <- testing[,which(Nas$isNA == 0)]

## Clean blancks

blancks<-which(ifelse(mapply(FUN=function(x){sum(x=="")},NewDF,SIMPLIFY = TRUE) >= 100,0,1)==0)
NewDF <- NewDF[,-blancks]

testingDF <- testingDF[,-blancks]

## remove user, X and time stamp part 1 until num window.

NewDF <- NewDF[,-c(1:7)]

testingDF <- testingDF[,-c(1:7)]
testingDF <- testingDF[,-53]

## Selecting cross validation method

par(mfcol=c(1,2))
plot(x=c(1:dim(NewDF)[1]),y = NewDF$total_accel_belt,col=NewDF$classe,xlab="observation",ylab="Total bel acceleration")
plot(x=c(1:dim(NewDF)[1]),y = NewDF$total_accel_arm,col=NewDF$classe,xlab="Observation",ylab="Total arm acceleration")

## Here we may see that the observations are split by order, so the technique we should use for slicing
## data should take random samples through the data frame to aleatorize the samples of the training set.

set.seed(123)
randomSamp <- sample(1:dim(NewDF)[1],size =dim(NewDF)[1],replace = FALSE)
NewDF<-NewDF[randomSamp,]
par(mfcol=c(1,2))
plot(x=c(1:dim(NewDF)[1]),y = NewDF$total_accel_belt,col=NewDF$classe,xlab="observation",ylab="Total bel acceleration")
plot(x=c(1:dim(NewDF)[1]),y = NewDF$total_accel_arm,col=NewDF$classe,xlab="Observation",ylab="Total arm acceleration")

## Before partitioning the data we will analyze correlation between variables

matrixCor <- abs(cor(NewDF[,-53]))
diag(matrixCor) <- 0
for(i in 1: dim(matrixCor)[1]){
  for(j in 1: dim(matrixCor)[1]){
    if(j > i){
      matrixCor[i,j] <- 0
    }
  }
}

highCor <- which(matrixCor > 0.8, arr.ind = T)

temporal <- data.frame(var1 =names(NewDF)[highCor[,1]], var2=names(NewDF)[highCor[,2]])

## Check variables with near zero variability

sum(nearZeroVar(NewDF[,-53],saveMetrics = TRUE)[,4])


#Now the data is not in order so we can take random samples or use kfold cross validation to perform our analysis

set.seed(123)

inTrain <- createDataPartition(NewDF$classe,p=0.7,list=FALSE)

myTrain <- NewDF[inTrain,]

myTest <- NewDF[-inTrain,]

## Create a model

# para ensayar

fitControl <- trainControl(method='cv', number = 3)

modelTrain <- train(classe ~.,data = myTrain,method="rpart", trControl= fitControl)
pred <- predict(modelTrain)
resultsTrain <- confusionMatrix(myTrain$classe,pred)
predTest <- predict(modelTrain,newdata=myTest)
resultsTest <- confusionMatrix(myTest$classe,predTest)
acRPART <- resultsTest[[3]][1]

rfModelTrain <- train(classe ~., data= myTrain,method="rf",trControl= fitControl,verbose=FALSE)
rfpred <- predict(rfModelTrain)
rfresultsTrain <- confusionMatrix(myTrain$classe,rfpred)
rfpredTest <- predict(rfModelTrain,newdata=myTest)
rfresultsTest <- confusionMatrix(myTest$classe,rfpredTest)
acRF <- rfresultsTest[[3]][1]

gbmModelTrain <- train(classe ~., data= myTrain,method="gbm",trControl= fitControl,verbose=FALSE)
gbmpred <- predict(gbmModelTrain)
gbmresultsTrain <- confusionMatrix(myTrain$classe,gbmpred)
gbmpredTest <- predict(gbmModelTrain,newdata=myTest)
gbmresultsTest <- confusionMatrix(myTest$classe,gbmpredTest)
acGBM <- gbmresultsTest[[3]][1]

resultSummary<- data.frame(Model = c("Predicting tree","Random Forest","Boosting with trees"),Accuracy=rbind(acRPART,acRF,acGBM))


## predicting validation Data with random forests

predictions <- predict(rfModelTrain,newdata = testingDF)
predictions
