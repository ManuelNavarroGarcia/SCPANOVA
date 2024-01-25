source(paste(getwd(), "/scpanova/R_utils.R", sep=""))
#improved_install_packages(c("devtools"))

#devtools::install_github("https://github.com/Victor-Espana/MLFrontiers.git")
library("aafs")

simulated_production_function_R <- function(data){
    # DEA
    start_time_DEA <- Sys.time()
    predictionsDEA <- aafs:::AAFS_BCC.OUT(nrow(data),
                                          as.matrix(data[, c(1, 2)]), 
                                          as.matrix(data[, 3]), 
                                          as.matrix(data[, c(1, 2)]), 
                                          as.matrix(data[, 3]), 
                                          2,
                                          1,
                                          nrow(data)) * data[, 3]
    end_time_DEA <- Sys.time()
    MAE_DEA <- mean(abs(predictionsDEA - data$yD))
    MSE_DEA <- mean((predictionsDEA - data$yD)^2)
    times_DEA <- difftime(end_time_DEA, start_time_DEA, units = "secs")

    # C2NLS
    start_time_C2NLS <- Sys.time()
    CCNLS <- C2NLS(data, c(1, 2), 3)
    predictionsCCNLS <- predict(CCNLS, data, c(1, 2))$f
    end_time_C2NLS <- Sys.time()

    MAE_C2NLS <- mean(abs(predictionsCCNLS - data$yD))
    MSE_C2NLS <- mean((predictionsCCNLS - data$yD)^2)
    times_C2NLS <- difftime(end_time_C2NLS, start_time_C2NLS, units = "secs")

    # AAFS
    start_time_AAFS <- Sys.time()
    model <- AAFS(data = data,
                  x = c(1, 2), 
                  y = 3,
                  nterms = 50,
                  Kp = 1,
                  d = 1,
                  err.red = 0.01,
                  minspan = -1,
                  endspan = -1,
                  knotsGrid = -1,
                  na.rm = TRUE)
    predictionsAAFS <- as.vector(predict(model, data, c(1, 2), 3))$y_pred
    end_time_AAFS <- Sys.time()

    MAE_AAFS <- mean(abs(predictionsAAFS - data$yD))
    MSE_AAFS <- mean((predictionsAAFS - data$yD)^2)
    times_AAFS <- difftime(end_time_AAFS, start_time_AAFS, units = "secs")

    return(c(MAE_DEA, MSE_DEA, times_DEA,
             MAE_C2NLS, MSE_C2NLS, times_C2NLS,
             MAE_AAFS, MSE_AAFS, times_AAFS))
}