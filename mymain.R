start_time = Sys.time()

train_main = read.csv('train.csv')
test_main = read.csv('test.csv')

test_main$Sale_Price = 0
train_main = train_main[!train_main$PID %in% c(902207130,910251050),]
test_main = test_main[!test_main$PID %in% c(902207130,910251050),]

data = rbind(train_main,test_main)

##################################Checking packages#############################
list.of.packages <- c("psych", "glmnet", "xgboost", "Rmisc",'plyr','dplyr','randomForest','e1071','gbm')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(plyr)
library(dplyr)
library(Rmisc)
library(randomForest)
library(psych)
library(xgboost)
library(e1071)
library(glmnet) 
library(gbm)


################################################################################
#Treatment and transformations
################################################################################
#Dropping outliers and unnecessary columns

drop = c('Garage_Yr_Blt','Utilities','Condition_2','X','Latitude','Longitude','Street')
data = data[,!(names(data)%in%drop)]



#converting to factor
data$MS_SubClass = as.factor(as.character(data$MS_SubClass))

#Converting ordinal variables to numeric by label encoding
x = levels(data$Lot_Shape)
data$Lot_Shape = as.numeric(as.character(mapvalues(data$Lot_Shape, from = x, to = c(1,2,4,3))))
x = levels(data$Land_Slope)
data$Land_Slope = as.numeric(as.character(mapvalues(data$Land_Slope, from = x, to = c(3,2,1))))
x = levels(data$Overall_Qual)
data$Overall_Qual = as.numeric(as.character(mapvalues(data$Overall_Qual, from = x, to = c(6,5,4,9,3,7,2,10,8,1))))
x = levels(data$Exter_Qual)
data$Exter_Qual = as.numeric(as.character(mapvalues(data$Exter_Qual, from = x, to = c(4,1,3,2))))
#x = levels(data$Exter_Cond)
#data$Exter_Cond = as.numeric(as.character(mapvalues(data$Exter_Cond, from = x, to = c(5,2,4,1,3))))
x = levels(data$Bsmt_Qual)
data$Bsmt_Qual = as.numeric(as.character(mapvalues(data$Bsmt_Qual, from = x, to = c(6,3,5,1,2,4))))
x = levels(data$Bsmt_Cond)
data$Bsmt_Cond = as.numeric(as.character(mapvalues(data$Bsmt_Cond, from = x, to = c(6,3,5,1,2,4))))
x = levels(data$Bsmt_Exposure)
data$Bsmt_Exposure = as.numeric(as.character(mapvalues(data$Bsmt_Exposure, from = x, to = c(4,5,3,2,1))))
x = levels(data$BsmtFin_Type_1)
data$BsmtFin_Type_1 = as.numeric(as.character(mapvalues(data$BsmtFin_Type_1, from = x, to = c(6,5,7,3,1,4,2))))
x = levels(data$BsmtFin_Type_2)
data$BsmtFin_Type_2 = as.numeric(as.character(mapvalues(data$BsmtFin_Type_2, from = x, to = c(6,5,7,3,1,4,2))))
x = levels(data$Heating_QC)
data$Heating_QC = as.numeric(as.character(mapvalues(data$Heating_QC, from = x, to = c(5,2,4,1,3))))
x = levels(data$Kitchen_Qual)
data$Kitchen_Qual = as.numeric(as.character(mapvalues(data$Kitchen_Qual, from = x, to = c(5,2,4,1,3))))
x = levels(data$Functional)
data$Functional = as.numeric(as.character(mapvalues(data$Functional, from = x, to = c(4,3,7,6,5,1,2,8))))
x = levels(data$Fireplace_Qu)
data$Fireplace_Qu = as.numeric(as.character(mapvalues(data$Fireplace_Qu, from = x, to = c(6,3,5,1,2,4))))
x = levels(data$Garage_Finish)
data$Garage_Finish = as.numeric(as.character(mapvalues(data$Garage_Finish, from = x, to = c(4,1,3,2))))
x = levels(data$Garage_Qual)
data$Garage_Qual = as.numeric(as.character(mapvalues(data$Garage_Qual, from = x, to = c(6,3,5,1,2,4))))
x = levels(data$Garage_Cond)
data$Garage_Cond = as.numeric(as.character(mapvalues(data$Garage_Cond, from = x, to = c(6,3,5,1,2,4))))
x = levels(data$Paved_Drive)
data$Paved_Drive = as.numeric(as.character(mapvalues(data$Paved_Drive, from = x, to = c(1,2,3))))
x = levels(data$Pool_QC)
data$Pool_QC = as.numeric(as.character(mapvalues(data$Pool_QC, from = x, to = c(5,2,4,1,3))))

x = levels(data$Fence)   #fence doesn't appear to have ordinal relationship
data$Fence = as.numeric(as.character(mapvalues(data$Fence, from = x, to = c(5,3,4,2,1))))

#Converting some discrete numeric variables to factor
data$Year_Sold = as.factor(data$Year_Sold)
data$Mo_Sold = as.factor(data$Mo_Sold)


####Converting rating to groups then converting to ordinal numeric - because of -ve correlation
first_cat = c('Very_Poor','Poor','Fair')
sec_cat = c('Below_Average','Average','Above_Average','Good','Very_Good','Excellent','Very_Excellent')
#third_cat = c('Very_Good','Excellent','Very_Excellent')
data$Overall_Cond = ifelse(data$Overall_Cond%in%first_cat, '1', ifelse(data$Overall_Cond%in%sec_cat, '2', '3'))
data$Overall_Cond = as.numeric(as.character(data$Overall_Cond))


######################################FEATURE ENGINEERING#################################################
PID = data$PID
drop = c('PID')
data = data[,!(names(data)%in%drop)]
#total sq ft
Total_SF = data$Total_Bsmt_SF + data$Gr_Liv_Area
data = cbind(Total_SF,data)
#bathrooms
Total_Bathrooms = data$Full_Bath + (data$Half_Bath*0.5) + data$Bsmt_Full_Bath + (data$Bsmt_Half_Bath*0.5)
data = cbind(Total_Bathrooms,data)
#joining back PID
data = cbind(PID,data)



####################################################################################################
#One hotting encoding and split
####################################################################################################

##one hot encoding
fake.y = rep(0, length(data[,1]))
tmp = model.matrix(~.,data = data)
tmp = data.frame(tmp[, -1])  # remove the 1st column (the intercept) of tmp
#write.csv(tmp,'tmp.csv')
data2 = tmp

##Removing predictors with less than 10 1's
fewOnes <- which(colSums(data2[1:nrow(data2),])<10)
drop = colnames(data2[fewOnes])
data2 = data2[,!(names(data2)%in%drop)]

#splitting data
train = data2[1:nrow(train_main),]
test = data2[(nrow(train_main)+1):nrow(data2),]


####################################################################################################
drop = c('PID')
train = train[,!(names(train)%in%drop)]

#transform sale price of train and other skewed variables
train$Sale_Price = log(train$Sale_Price)
train$Gr_Liv_Area = log(train$Gr_Liv_Area)
train$First_Flr_SF = log(train$First_Flr_SF)
train$Mas_Vnr_Area = log(1+train$Mas_Vnr_Area)
train$Open_Porch_SF = log(1+train$Open_Porch_SF)
train$Wood_Deck_SF = log(1+train$Wood_Deck_SF)
train$Lot_Area = log(1+train$Lot_Area)
train$Pool_Area = log(1+train$Pool_Area)
train$Three_season_porch = log(1+train$Three_season_porch)
train$Low_Qual_Fin_SF = log(1+train$Low_Qual_Fin_SF)
train$BsmtFin_SF_2 = log(1+train$BsmtFin_SF_2)
train$Screen_Porch = log(1+train$Screen_Porch)
train$Enclosed_Porch = log(1+train$Enclosed_Porch)
train$Total_SF = log(train$Total_SF)
train$Bsmt_Unf_SF = log(1+train$Bsmt_Unf_SF)



#Winsorize
winsorize = function(vec,pr1,pr2){
  x = quantile(vec,probs=c(pr1,pr2))
  vec[vec>x[2]] = x[2]
  vec[vec<x[1]] = x[1]
  return(vec)
}

train$Gr_Liv_Area = winsorize(train$Gr_Liv_Area,0.005,0.99)
train$Total_Bsmt_SF = winsorize(train$Total_Bsmt_SF,0,0.995)
train$First_Flr_SF = winsorize(train$First_Flr_SF,0.003,0.997)
train$Garage_Area = winsorize(train$Garage_Area,0,0.997)
train$Mas_Vnr_Area = winsorize(train$Mas_Vnr_Area,0,0.997)
train$Total_SF = winsorize(train$Total_SF,0.005,0.995)
train$Sale_Price = winsorize(train$Sale_Price,0.01,0.99)


#Removing outlier
train = train[!(train$Sale_Price>12.25 & train$Total_SF>1000 & train$Total_SF<1500),]


#Modify test data
drop = c('Sale_Price')
test = test[,!(names(test)%in%drop)]
test$Gr_Liv_Area = log(test$Gr_Liv_Area)
test$First_Flr_SF = log(test$First_Flr_SF)
test$Mas_Vnr_Area = log(1+test$Mas_Vnr_Area)
test$Open_Porch_SF = log(1+test$Open_Porch_SF)
test$Wood_Deck_SF = log(1+test$Wood_Deck_SF)
test$Lot_Area = log(1+test$Lot_Area)
test$Pool_Area = log(1+test$Pool_Area)
test$Three_season_porch = log(1+test$Three_season_porch)
test$Low_Qual_Fin_SF = log(1+test$Low_Qual_Fin_SF)
test$BsmtFin_SF_2 = log(1+test$BsmtFin_SF_2)
test$Screen_Porch = log(1+test$Screen_Porch)
test$Enclosed_Porch = log(1+test$Enclosed_Porch)
test$Total_SF = log(test$Total_SF)
test$Bsmt_Unf_SF = log(1+test$Bsmt_Unf_SF)

#Creating matrices and fitting models
X_train = data.matrix(train[,1:(length(train)-1)])
Y_train = data.matrix(train[,ncol(train)])
X_test = data.matrix(test[,2:length(test)]) 


#LASSO
lam.seq = exp(seq(-15, 15, length=100))
cv.out = cv.glmnet(X_train, Y_train, lambda = lam.seq, nfolds = 200)
best.lam = cv.out$lambda.min
Ytest.pred.lasso = predict(cv.out, s = best.lam, newx = X_test)
mylasso.coef = predict(cv.out, s = best.lam, type = "coefficients")
model_size = sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min


#XGBOOST
gbm.fit.final <- gbm(
  formula = Sale_Price ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 483,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.minobsinnode = 5,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)
Ytest.pred.xg = predict(gbm.fit.final, test[,2:length(test)], n.trees = 483)


#RANDOM FOREST
rfModel = randomForest(Sale_Price ~ ., data = train,importance = T, ntree=450); 
Ytest.pred.rf = predict(rfModel, X_test)






#OUTPUT 1
output = data.frame(test$PID,exp(Ytest.pred.lasso))
colnames(output) = c('PID','Sale_Price')
write.csv(output,'mysubmission1.txt',row.names = FALSE)

#OUTPUT 2
output = data.frame(test$PID,exp(Ytest.pred.xg)) #CHANGEEEEE
colnames(output) = c('PID','Sale_Price')
write.csv(output,'mysubmission2.txt',row.names = FALSE)


#EVALUATION METRIC
# test.y = read.csv('test_pidsale.csv')###################remove?
# pred = read.csv("mysubmission1.txt")
# names(test.y)[2] = "True_Sale_Price"
# pred = merge(pred, test.y, by="PID")
# sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))


end_time = Sys.time()
end_time-start_time
