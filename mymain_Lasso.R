start_time = Sys.time()

train_main = read.csv('train.csv')
test_main = read.csv('test.csv')

test_main$Sale_Price = 0
train_main = train_main[!train_main$PID %in% c(902207130,910251050),]
test_main = test_main[!test_main$PID %in% c(902207130,910251050),]

data = rbind(train_main,test_main)

##################################Checking packages#############################
list.of.packages <- c("psych", "glmnet", "xgboost", "Rmisc",'plyr','dplyr','randomForest','e1071','gbm','tibble')
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
library(tibble)

################################################################################
#Treatment and transformations
################################################################################
#Dropping outliers and unnecessary columns

drop = c('Garage_Yr_Blt','Utilities','Condition_2','X','Latitude','Longitude','Street','Heating','Roof_Matl',
         'Land_Slope','Pool_QC','Misc_Feature', 'Low_Qual_Fin_SF','Three_season_porch',
         'Pool_Area','Misc_Val','Garage_Area','TotRms_AbvGrd','Alley','Mas_Vnr_Area')
data = data[,!(names(data)%in%drop)]

data$MS_SubClass = as.factor(as.character(data$MS_SubClass))
data$Mo_Sold = as.factor(as.character(data$Mo_Sold))
data$Year_Sold = as.factor(as.character(data$Year_Sold))



#FEATURE ENGINEERING
Total_SF = data$Total_Bsmt_SF + data$Gr_Liv_Area
data = add_column(data, Total_SF, .after = "PID")

Total_Bathrooms = data$Full_Bath + data$Half_Bath*0.5 + data$Bsmt_Full_Bath + data$Bsmt_Half_Bath*0.5
data = add_column(data, Total_Bathrooms, .after = "PID")

Remod_yesno = ifelse(data$Year_Built == data$Year_Remod_Add, 0, 1)
data = add_column(data, Remod_yesno, .after = "PID")

Age = as.numeric(as.character(data$Year_Sold))-data$Year_Remod_Add
data = add_column(data, Age, .after = "PID")

IsNew = ifelse(data$Year_Sold==data$Year_Built, 1, 0)
data = add_column(data, IsNew, .after = "PID")



##one hot encoding
tmp = model.matrix(~.,data = data)
tmp = data.frame(tmp[, -1])  # remove the 1st column (the intercept) of tmp
data2 = tmp

#splitting data
train = data2[1:nrow(train_main),]
test = data2[(nrow(train_main)+1):nrow(data2),]

#Log transformations of train data
train$Lot_Area = log(train$Lot_Area)
train$BsmtFin_SF_2 = log(1+train$BsmtFin_SF_2)
train$Sale_Price = log(train$Sale_Price)
train$Total_SF = log(train$Total_SF)

#winsorizing train data
winsorize = function(vec,pr1,pr2){
  x = quantile(vec,probs=c(pr1,pr2))
  vec[vec>x[2]] = x[2]
  vec[vec<x[1]] = x[1]
  return(vec)
}
train$Gr_Liv_Area = winsorize(train$Gr_Liv_Area,0.005,0.99)
train$Total_Bsmt_SF = winsorize(train$Total_Bsmt_SF,0,0.995)
train$First_Flr_SF = winsorize(train$First_Flr_SF,0.003,0.997)
#train$Garage_Area = winsorize(train$Garage_Area,0,0.997)
#train$Mas_Vnr_Area = winsorize(train$Mas_Vnr_Area,0,0.997)
train$Total_SF = winsorize(train$Total_SF,0.005,0.995)
train$Sale_Price = winsorize(train$Sale_Price,0.01,0.99)

#Drop pid
drop = c('PID')
train = train[,!(names(train)%in%drop)]

###MODIFIY TEST DATA
test$Lot_Area = log(1+test$Lot_Area)
test$BsmtFin_SF_2 = log(1+test$BsmtFin_SF_2)
test$Total_SF = log(test$Total_SF)

drop = c('Sale_Price')
test = test[,!(names(test)%in%drop)]

##################################################################################################################
#LASSO ALGORITHM
one_step_lasso = function(r, x, lam){
  xx = sum(x^2)
  xr = sum(r*x)
  b = (abs(xr) -lam/2)/xx
  b = sign(xr)*ifelse(b>0, b, 0)
  return(b)
}

mylasso = function(X, y, lam, n.iter = 50, standardize  = TRUE)
{
  p = ncol(X)
  b = rep(0, p)
  X_mean = rep(0, p)
  y_mean = mean(y)

  
  for(i in 1:p){
    X_mean[i] = mean(X[,i])
    X[,i] = (X[,i] - X_mean[i])
  }
  # Centering y
  y = y - mean(y)
  for(step in 1:n.iter){
    r = y - X%*%b
    for(j in 1:p){
      r = r + X[, j] * b[j]
      b[j] = one_step_lasso(r, X[, j], lam)
      r = r - X[, j] * b[j]
    }
  }
  num_sum = 0
  for(j in 1:p){
    num_sum = num_sum + b[j]*X_mean[j]
  }
  b0 = y_mean - num_sum
  return(c(b0, b))
}
###################################################################################################################

#data
X_train = data.matrix(train[,1:(length(train)-1)])
Y_train = data.matrix(train[,ncol(train)])
X_test = data.matrix(test[,2:length(test)]) 

###################################################################################################################
#optimum lambda value using glm net

lam.seq = exp(seq(-15, 10, length=100))
cv.out = cv.glmnet(X_train, Y_train, lambda = lam.seq, nfolds = 200)
best.lam = cv.out$lambda.min

######################################################
vec1 = mylasso(X_train, Y_train, lam = best.lam, n.iter = 300)
#vec1 = mylasso(X_train, Y_train, lam = (2*nrow(X_train)*39), n.iter = 300)
b0 = vec1[1]
b = vec1[-1]

Ytest.pred = b0 + X_test %*% b
Ytest.pred = exp(Ytest.pred)

#OUTPUT 
output = data.frame(test$PID,Ytest.pred)
colnames(output) = c('PID','Sale_Price')
write.csv(output,'mysubmission3.txt',row.names = FALSE)


#EVALUATION METRIC
# test.y = read.csv('test_pidsale.csv')###################remove?
# pred = read.csv("mysubmission3.txt")
# names(test.y)[2] = "True_Sale_Price"
# pred = merge(pred, test.y, by="PID")
# sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))




end_time = Sys.time()
end_time-start_time

