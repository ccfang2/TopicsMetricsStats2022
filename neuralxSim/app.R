# ***********************************
# Shiny R App -- Bauer and Kohler (2019)
# ***********************************

# This is a Shiny R App that helps to visualize the prediction accuracy of multilayer
# hierarchical neural networks proposed by Bauer and Kohler (2019) and compare it 
# with the prediction accuracy of some alternative models

rm(list = ls())
library(shiny)
library(FNN)
library(neuralnet)
library(RSNNS)
library(randomForestSRC)
library(ggplot2)
library(patchwork)
library(tidyverse)
library(ggthemes)
library(ggcorrplot)

# define UI for app
ui <- fluidPage(
  #app title
  titlePanel("Multilayer Hierarchical Neural Network Simulation"),
  sidebarLayout(
    sidebarPanel(
      p("This is an app that helps to visualze the prediction accuracy of multilayer
        hierarchical neural networks and compare it with some alternative models in
        different cases"),
      selectInput("fct","Select a function for data generation",
                  choices = list("m1"=1,"m2"=2,"m4"=4,"m5"=5),
                  selected = 4),
      helpText("m1 and m2: ordinary general hierarchical interaction functions"), 
      helpText("m4: additive function with d*=1"), 
      helpText("m5: interaction function with d*=d"),
      br(),
      radioButtons("noise", "Select the degree of disturbance noise",
                   choices = list("5 %" = 1, "20%" = 2),
                   selected = 1),
      br(),
      checkboxGroupInput("model","Select all models you want to compare",
                         choices = list("knn"=1,
                                        "rf"=2,
                                        "neural1"=3,
                                        "neuralx"=4),
                         selected = 1),
      helpText("knn: simple nearest neighbor estimate"),
      helpText("rf: random forest"), 
      helpText("neural1: fully connected neural network with one hidden layer"), 
      helpText("neuralx: multilayer hierarchical neural network"),
      # Input: actionButton() to defer the rendering of output ----
      # until the user explicitly clicks the button (rather than
      # doing it immediately when inputs change). This is useful if
      # the computations required to render output are inordinately
      # time-consuming.
      actionButton("update", "Run")
    ),
    # main panel to display outputs
    mainPanel(
      plotOutput("plots", width = "100%",
                 height="800px")
    )
  )
)

# define server logic for app
server <- function (input, output) {
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
  
  mlist <- list(m1,m2,m3,m4,m5,m6)
  
  # Define coefficients
  lambda <- c(9.11, 5.68, 13.97, 1.77, 1.64, 2.47)
  sigma <- c(0.05, 0.2)
  
  # Define train data generation function
  train.gen <- function(n,i,j) {
    # n: number of observations in train set, m: model, 
    # i: model index, j: sigma index
    m <- mlist[[i]]
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
  test.gen <- function(N,i,j) {
    # N: number of observations in test set, m: model, 
    # i: model index, j: sigma index
    m <- mlist[[i]]
    ds <- c(7,7,10,7,7,7)
    d<- ds[i]
    Xmat <- matrix(runif(d*N),N,d)
    epsilon <- rnorm(N)
    Xe <- cbind(Xmat,epsilon)
    Y <- apply(Xe,1,function(X) m(X[1:d])+sigma[j]*lambda[i]*X[d+1])
    data <- data.frame(Y,Xe[,-(d+1)])
    return(data)
  }
  
  # Data Generation Process
  dataInput <- eventReactive(input$update,{
    train <- train.gen(n=100,i=as.numeric(input$fct),j=as.numeric(input$noise))
    test <- test.gen(N=200,i=as.numeric(input$fct),j=as.numeric(input$noise))
    return(list(train, test))
  })
  
  
  output$plots <- renderPlot({
    
    dataset <- dataInput()
    train <- dataset[[1]]
    test <- dataset[[2]]
    
    # Dimension of input
    d <- ncol(train[,-1])
    
    # KNN Model
    knn.model <- knn.reg(train[,-1],test[,-1],train[,1],k=4)
    knn.pred <- as.matrix(knn.model$pred,ncol=1)
    colnames(knn.pred) <- "knn"
    knn.error <- as.matrix(test[,1]-knn.pred,ncol=1)
    colnames(knn.error) <- "knn"
    knn.L2 <- mean((knn.model$pred-test[,1])^2)
    
    # RBF Model
    # rbf.model <- rbf(as.matrix(train[,-1]),as.matrix(train[,1]))
    # rbf.pred <- predict(rbf.model,newdata= test[,-1])
    # rbf.error <- test[,1]-rbf.pred
    # rbf.L2 <- mean((rbf.pred-test[,1])^2)
    
    # Random Forest Model
    rf.model <- rfsrc(Y~., train)
    rf.pred.set <- predict(rf.model, test[,-1])
    rf.pred <- as.matrix(rf.pred.set$predicted,ncol=1)
    colnames(rf.pred)<-"rf"
    rf.error <- as.matrix(test[,1]-rf.pred,ncol=1)
    colnames(rf.error)<-"rf"
    rf.L2 <- mean((rf.pred-test[,1])^2)
    
    # Neural-1 Model
    neural1.model <- neuralnet(Y~., train, hidden=5, 
                               threshold=0.05, stepmax=1e+07)
    neural1.pred <- predict(neural1.model,test[,-1])
    colnames(neural1.pred)<-"neural1"
    neural1.error <- test[,1]-neural1.pred
    neural1.L2 <- mean((neural1.pred-test[,1])^2)
    
    # Neural-X Model (l=0)
    null <- (d+1)*24+c(10:25,27:34,43:50,52:67)
    
    neuralx.model <- neuralnet(Y~., train, hidden=c(24,3),
                               threshold=0.05, stepmax=1e+08,exclude=null,
                               constant.weights = rep(1e-17,length(null)))
    neuralx.pred <- predict(neuralx.model,test[,-1])
    colnames(neuralx.pred)<-"neuralx"
    neuralx.error <- test[,1]-neuralx.pred
    neuralx.L2 <- mean((neuralx.pred-test[,1])^2)
    
    # Base Model
    L2.base.sim <- replicate(10, {
      train.base <- train.gen(n=100,i=as.numeric(input$fct),j=1)
      L2.base <- mean((mean(train.base[,1])-test[,1])^2)
    })
    L2.avg <- median(L2.base.sim)
    
    # results of all models
    errors <- cbind(knn.error, rf.error, neural1.error, neuralx.error)
    L2 <- c(knn.L2/L2.avg, rf.L2/L2.avg,
                neural1.L2/L2.avg, neuralx.L2/L2.avg)
   # names(L2) <- c("knn","rf","neural1","neuralx")
    preds <- cbind(knn.pred, rf.pred, neural1.pred, neuralx.pred)
    names <- c("knn","rf","neural1","neuralx")
      
    # models selected
    selected <- as.numeric(input$model)
    names.selected <- names[selected]
    
    errors.selected <- cbind(errors[,selected],v=1:nrow(test))
    L2.selected <- cbind(mse=L2[selected],v=1:length(selected))
    preds.selected <- cbind(true.value=test[,1],preds[,selected])
    
    
    # plots
    plot1 <- errors.selected %>% 
      as.data.frame %>% 
      pivot_longer(-v) %>%
      ggplot(aes(x=factor(v),y=value,color=name,group=name))+
      geom_line()+
      labs(title = "Prediction Error in Test Data", x="",y="Error")+
      scale_color_discrete("Model")+
      theme_bw()
    
    plot2 <- L2.selected %>%
      as.data.frame %>% 
      ggplot(aes(x=factor(v)))+
      geom_bar(aes(weight=mse))+
      labs(title = "Scaled Empirical L2 Error", x="",y= "")+
      scale_x_discrete(labels = names.selected)+
      theme_bw()
    
    plot3<- cor(preds.selected) %>%
      ggcorrplot(method = "circle", type = "lower", show.diag = T,
                 legend.title = "", title ="Correlation Matrix")
    
    plot1 /(plot2|plot3)
  })
}

# create shiny app
shinyApp(ui, server)