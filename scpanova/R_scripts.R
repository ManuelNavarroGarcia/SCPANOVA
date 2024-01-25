source(paste(getwd(), "/scpanova/R_utils.R", sep=""))
improved_install_packages(c("scam", "cgam", "glue", "R.utils", "LaplacesDemon"))

suppressMessages(suppressWarnings(library(scam)))
suppressMessages(suppressWarnings(library(cgam)))
suppressMessages(suppressWarnings(library(glue)))
suppressMessages(suppressWarnings(library(R.utils)))
suppressMessages(suppressWarnings(library(LaplacesDemon)))

simulated_example_results_R <- function(df, n_knots, family, scenario){
  if (family == "gaussian"){
    family_ <- gaussian()
  } else if (family == "poisson") {
     family_ <- poisson(link="log")
  } else {
    family_ <- binomial(link="logit")
  }

  start_time_scam <- Sys.time()
  if (scenario == 2){
    bs_ <- "tecxcx"
  } else if (scenario == 3) {
     bs_ <- "temicv"
  } else {
    bs_ <- "tedmi"
  }

  print("Working with scam package")
  if (scenario == 3){
    b_scam <- scam(y~s(x2, x1, k=c(n_knots, n_knots), bs=bs_, m=c(2, 2)),
                   data=list(x1=df$x1, x2=df$x2, y=df$y_error),
                   family=family_)
  } else {
    b_scam <- scam(y~s(x1, x2, k=c(n_knots, n_knots), bs=bs_, m=c(2, 2)),
                   data=list(x1=df$x1, x2=df$x2, y=df$y_error),
                   family=family_)
  }
  pred_scam <- b_scam$linear.predictors
  end_time_scam <- Sys.time()

  MAE_theo_scam <- mean(abs(pred_scam - df$y_))
  MSE_theo_scam <- mean((pred_scam - df$y_)^2)
  times_scam <- difftime(end_time_scam, start_time_scam, units = "secs")
  
  print("Working with cgam package")
  start_time_cgam <- Sys.time()
  if (scenario == 2) {
     b_cgam <- cgam(y ~ s.conv.conv(x1, x2, numknots = c(n_knots, n_knots),
                           space = c("E", "E")),
                      gcv=T,
                      family=family_,
                      data=list(x1=df$x1, x2=df$x2, y=df$y_error))
  } else if (scenario == 3) {
    # This does not count as `cgam` cannot address tensor product of concave and
    # increasing surfaces
    b_cgam <- cgam(y ~ s.incr.incr(x1, x2, numknots = c(5, 5),
                           space = c("E", "E")),
                      gcv=T,
                      family=family_,
                      data=list(x1=df$x1, x2=df$x2, y=df$y_error))
  } else {
    b_cgam <- cgam(y ~ s.incr.incr(x1, x2, numknots = c(n_knots, n_knots),
                          space = c("E", "E")),
                     gcv=T,
                     family=family_,
                     data=list(x1=df$x1, x2=df$x2, y=df$y_error))
  }
  if (family == "gaussian"){
    pred_cgam <- b_cgam$muhat
  } else if (family == "poisson") {
    pred_cgam <- log(b_cgam$muhat)
  } else {
    pred_cgam <- log(b_cgam$muhat / (1 - b_cgam$muhat))
  }
  end_time_cgam <- Sys.time()
  
  MAE_theo_cgam <- mean(abs(pred_cgam - df$y_))
  MSE_theo_cgam <- mean((pred_cgam - df$y_)^2)
  times_cgam <- difftime(end_time_cgam, start_time_cgam, units = "secs")
  return(c(MAE_theo_scam,
           MSE_theo_scam,
           times_scam,
           MAE_theo_cgam,
           MSE_theo_cgam,
           times_cgam))
}

scam_hschool <- function(train, test){
  train <- data.frame(math = as.numeric(train$math),
                      langarts = as.numeric(train$langarts),
                      daysabs = as.numeric(train$daysabs))

  test <- data.frame(math = as.numeric(test$math),
                     langarts = as.numeric(test$langarts),
                     daysabs = as.numeric(test$daysabs))

  start_time_scam <- Sys.time()
  # Double monotone decreasing P-splines bs="tedmd"
  b_scam <- scam(daysabs~s(math, langarts, k=c(13 + 3, 13 + 3), bs="tedmd", m=c(2, 2)),
                 data=train,
                 family=poisson(link="log"))
  # Inverse transformation of the linear predictor
  pred_scam <- exp(predict(b_scam, test))
  end_time_scam <- Sys.time()
  times_scam <- difftime(end_time_scam, start_time_scam, units = "secs")

  MAE_scam <- mean(abs(pred_scam-test$daysabs))
  MSE_scam <- mean((pred_scam-test$daysabs)^2)
  return(c(toString(MAE_scam), toString(MSE_scam), toString(times_scam)))
}

cgam_hschool <- function(train, test, n_knots){
  train <- data.frame(math = as.numeric(train$math),
                      langarts = as.numeric(train$langarts),
                      daysabs = as.numeric(train$daysabs))

  test <- data.frame(math = as.numeric(test$math),
                     langarts = as.numeric(test$langarts),
                     daysabs = as.numeric(test$daysabs))

  start_time_cgam <- Sys.time()
  # Double monotone decreasing surface with s.decr.decr
  b_cgam <- cgam(daysabs ~ s.decr.decr(math, 
                                       langarts,
                                       numknots = c(n_knots, n_knots),
                                       space = c("E", "E")), 
                 gcv=T,
                 data=train,
                 family=poisson(link="log"))
  pred_cgam <- predict(b_cgam, test[c("math", "langarts")])$fit
  end_time_cgam <- Sys.time()
  times_cgam <- difftime(end_time_cgam, start_time_cgam, units = "secs")

  MAE_cgam <- mean(abs(pred_cgam-test$daysabs))
  MSE_cgam <- mean((pred_cgam-test$daysabs)^2)
  return(c(toString(MAE_cgam), toString(MSE_cgam), toString(times_cgam)))
}

scam_additive_pima <- function(train, test, k){
  train <- data.frame(col1 = as.numeric(train[, 1]),
                      col2 = as.numeric(train[, 2]),
                      col3 = as.numeric(train[, 3]))

  test <- data.frame(col1 = as.numeric(test[, 1]),
                     col2 = as.numeric(test[, 2]))

  start_time_pya <- Sys.time()
  # Double monotone increasing P-splines bs="mpi"
  b_pya <- scam(col3 ~ s(col1, k=k, bs="mpi", m=2) + s(col2, k=k, bs="mpi", m=2),
                data=train,
                family=binomial(link="logit"))
  end_time_pya <- Sys.time() 
  pred_pya <- invlogit(predict(b_pya, test))
  times_pya <- difftime(end_time_pya, start_time_pya, units = "secs")

  return(c(toString(pred_pya), toString(times_pya)))
}

scam_interaction_pima <- function(train, test){
  train <- data.frame(col1 = as.numeric(train[, 1]),
                      col2 = as.numeric(train[, 2]),
                      col3 = as.numeric(train[, 3]))

  test <- data.frame(col1 = as.numeric(test[, 1]),
                     col2 = as.numeric(test[, 2]))

  start_time_pya <- Sys.time() 
  # Double monotone increasing P-splines bs="tedmi"  
  b_pya <- scam(col3 ~ s(col1, col2, k=c(23 + 3, 23 + 3), bs="tedmi", m=c(2, 2)),
                data=train,
                family=binomial(link="logit")) 
  end_time_pya <- Sys.time()  
  pred_pya <- invlogit(predict(b_pya, test))
  times_pya <- difftime(end_time_pya, start_time_pya, units = "secs")

  return(c(toString(pred_pya), toString(times_pya)))
}