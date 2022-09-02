
# ***********************************
# Simulations -- Bauer and Kohler (2019)
# ***********************************
rm(list = ls())

# Libraries
library(FNN)
library(neuralnet)
library(RSNNS)
library(randomForestSRC)

# -----------------------------------
# Data Preparation
# -----------------------------------

# Define models
m1 <- function(X) {
  result <- 1/tan(pi/(1+exp(X[1]^2+2*X[2]+sin(6*X[4]^3)-3)))+
    exp(3*X[3]+2*X[4]-5*X[5]+sqrt(X[6]+0.9*X[7]+0.1))
  return(result)
}

m2 <- function(X) {
  result <- 2/(X[1]+0.008)+3*log(X[2]^7*X[3]+0.1)*X[4]
  return(result)
}

m3 <- function(X) {
  result <- 2*log(X[1]*X[2]+4*X[3]+abs(tan(X[4]))+0.1)+
    X[3]^4*X[5]^2*X[6]-X[4]*X[7]+
    (3*X[8]^2+X[9]+2)^(0.1+4*X[10]^2)
  return(result)
}

m4 <- function(X) {
  result <- X[1]+tan(X[2])+X[3]^3+log(X[4]+0.1)+3*X[5]+
    X[6]+sqrt(X[7]+0.1)
  return(result)
}

m5 <- function(X) {
  result <- exp(sqrt(sum(X^2)))
  return(result)
}

m6 <- function(X) {
  OMatrix <- randomOrthogonalMatrix(7,7, seed = 819)
  result <- m1(1/2*abs(OMatrix)%*%X)
}


# Define coefficients
lambda <- c(9.11, 5.68, 13.97, 1.77, 1.64, 2.47)
sigma <- c(0.05, 0.2)

# Define train data generation function
train.gen <- function(n,m,i,j) {
  # n: number of observations in train set, m: model, 
  # i: model index, j: sigma index
  ds <- c(7,7,10,7,7,7)
  d <- ds[i]
  Xmat <- matrix(runif(d*n),n,d)
  epsilon <- rnorm(n)
  Xe <- cbind(Xmat,epsilon)
  Y <- apply(Xe,1,function(X) m(X[1:d])+sigma[j]*lambda[i]*X[d+1])
  data <- data.frame(Y,Xe[,-(d+1)])
  return(data)
}

# Define test data generation function
test.gen <- function(N,m,i,j) {
  # N: number of observations in test set, m: model, 
  # i: model index, j: sigma index
  ds <- c(7,7,10,7,7,7,14)
  d<- ds[i]
  Xmat <- matrix(runif(d*N),N,d)
  epsilon <- rnorm(N)
  Xe <- cbind(Xmat,epsilon)
  Y <- apply(Xe,1,function(X) m(X[1:d])+sigma[j]*lambda[i]*X[d+1])
  data <- data.frame(Y,Xe[,-(d+1)])
  return(data)
}

# -----------------------------------
# Prediction Models
# -----------------------------------
set.seed(819)

L2.fct <- function(n,N,m,i,j,k){
  # n: number of observations in train set,
  # N: number of observations in test set, m: model, 
  # i: model index, j: sigma index, 
  # k: number of neighbors considered
  train <- train.gen(n,m,i,j)
  test <- test.gen(N,m,i,j)
  
  # Dimension of input
  d <- ncol(train[,-1])
  
  # KNN Model
  knn.model <- knn.reg(train[,-1],test[,-1],train[,1],k)
  knn.L2 <- mean((knn.model$pred-test[,1])^2)
  
  # RBF Model
  rbf.model <- rbf(train[,-1],train[,1])
  rbf.pred <- predict(rbf.model,test[,-1])
  rbf.L2 <- mean((rbf.pred-test[,1])^2)
  
  # Random Forest Model
  rf.model <- rfsrc(Y~., train)
  rf.pred <- predict(rf.model, test[,-1])
  rf.L2 <- mean((rf.pred$predicted-test[,1])^2)
  
  # Neural-1 Model
  neural1.model <- neuralnet(Y~., train, hidden=5, 
                             threshold=0.05, stepmax=1e+07)
  neural1.pred <- predict(neural1.model,test[,-1])
  neural1.L2 <- mean((neural1.pred-test[,1])^2)
  
  # Neural-X Model (l=0)
  null <- (d+1)*24+c(10:25,27:34,43:50,52:67)

  neuralx.model <- neuralnet(Y~., train, hidden=c(24,3),
                             threshold=0.05, stepmax=1e+08,exclude=null,
                             constant.weights = rep(1e-17,length(null)))
  neuralx.pred <- predict(neuralx.model,test[,-1])
  neuralx.L2 <- mean((neuralx.pred-test[,1])^2)
  
  # Base Model
  L2.base.sim <- replicate(10, {
    train.base <- train.gen(n,m,i,j)
    L2.base <- mean((mean(train.base[,1])-test[,1])^2)
    })
  L2.avg <- median(L2.base.sim)
  
  return(c(knn.L2/L2.avg, rbf.L2/L2.avg, rf.L2/L2.avg,
           neural1.L2/L2.avg, neuralx.L2/L2.avg))
}

#, neuralx.L2/L2.avg

sim <- function(n,N=500,m,j,k=4) {
  i <- as.numeric(substr(as.character(substitute(m)),2,2))
  rep <- replicate(10,L2.fct(n,N,m,i,j,k))
  return(cbind(apply(rep,1,median),apply(rep,1,IQR)))
}

# -----------------------------------
# Prediction Results
# -----------------------------------
n <- c(100,200)
j <- 1
# m1
sim_m1 <- lapply(n, function(x) lapply(j, function(y) sim(n=x,m=m1,j=y)))

sim(n=100,m=m1,j=1)

sim(n=100,m=m2,j=1)
sim(n=100,m=m2,j=2)

sim(n=100,m=m4,j=1)
sim(n=100,m=m4,j=2)

sim(n=100,m=m5,j=1)
sim(n=100,m=m5,j=2)

sim(n=100,m=m6,j=1)
sim(n=100,m=m6,j=2)

sim(n=100,m=m5,j=1)
sim(n=100,m=m5,j=2)
sim(n=100,m=m2,j=2)

Sys.time()
sim(n=100,m=m3,j=1)
Sys.time()


