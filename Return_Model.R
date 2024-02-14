# clean the environment and console before running the code
rm(list = ls())

# Load the data
df <- read.csv("mpd_sp500_lagged.csv")

# view the columns 
names(df)

# box jenkins method
# install.packages("forecast")
library(forecast)

sum(is.na(df))

# drop NA values
df <- na.omit(df)

# check na values
sum(is.na(df))


# Define features and target variable
start_col_index <- which(names(df) == "f1_mu_lag1")
end_col_index <- which(names(df) == "SP_lg_pr_lag6")
column_indices <- start_col_index:end_col_index
X <- df[, column_indices]
y <- df$SP_lg_pr  # Target variable (log return)

# # Split the data into training and testing sets
# set.seed(12345)
# train_indices <- sample(1:nrow(df), 0.75 * nrow(df))
# X_train <- X[train_indices, ]
# X_test <- X[-train_indices, ]
# y_train <- y[train_indices]
# y_test <- y[-train_indices]

# chronological order
# Calculate the index at which to split the dataset (75% for training, 25% for testing)
split_index <- round(0.75 * nrow(df))

# Split the dataset while preserving temporal order
X_train <- X[1:split_index, ]
X_test <- X[(split_index + 1):nrow(df), ]
y_train <- y[1:split_index]
y_test <- y[(split_index + 1):nrow(df)]



# do autoregression on log return and its lagged values from lag 1 to lag 6
# Assuming 'y' is your log return series
library(forecast)

# Assuming forecast library is already loaded
# library(forecast)

# Fit AR models for lags 1 to 6 and compare AIC and BIC
results <- data.frame(Lag = integer(), AIC = double(), BIC = double())

for (lag in 1:6) {
  # Fit AR model with 'lag' lags
  model <- arima(y_train, order=c(lag,1,0))
  
  # Append the lag, AIC, and BIC to the results dataframe
  results <- rbind(results, data.frame(Lag = lag, AIC = AIC(model), BIC = BIC(model)))
}

# Print the results
print(results)

# Optionally, to find the model with the lowest AIC and BIC
best_by_aic <- results[which.min(results$AIC),]
best_by_bic <- results[which.min(results$BIC),]

print(paste("Best model by AIC: AR", best_by_aic$Lag))
print(paste("Best model by BIC: AR", best_by_bic$Lag))

# CROSS VALIDATION LASSO with cv=10
# install.packages("glmnet")
library(Matrix)
library(glmnet)

# Assuming 'X' and 'y' are your predictors and response variable, respectively
# Ensure 'X' is a matrix
X_matrix <- as.matrix(X_train)
Y <- as.vector(y_train)

# Perform 10-fold cross-validation for Lasso
set.seed(12345) # For reproducibility
l_cv_model <- cv.glmnet(X_matrix, Y, alpha = 1, nfolds = 10)

# Extract the best lambda value
l_cv_best_lambda <- l_cv_model$lambda.min
cat("lassoCV Best lambda: ", l_cv_best_lambda, "\n")

# Fit the final model using the best lambda
lasso_CV <- glmnet(X_matrix, Y, alpha = 1, lambda = l_cv_best_lambda)

lasso_CV_coef <- as.vector(coef(lasso_CV))
# Extract the number of non-zero coefficients
l_cv_non_zero_coef <- (lasso_CV_coef[lasso_CV_coef!=0])[-1] #Subtract 1 for the intercept

cat("Lasso CV Number of non-zero coefficients: ", length(l_cv_non_zero_coef), "\n") 

####################################################################################################################################
# ADAPTIVE LASSO (需要调)
# do an adaptive Lasso model
# Fit the adaptive Lasso model
 
set.seed(12345)

# Step 1: Fit an initial Lasso model
initial_a_lasso <- glmnet(X_matrix, Y, alpha = 1)

# Step 2: Calculate weights. Avoid division by zero or extremely small coefficients.
epsilon_a_lasso <- 1e-4  # Small number to avoid division by zero
# initial_coef_a_lasso <- coef(initial_a_lasso, initial_a_lasso$lambda[1])[,1]
initial_coef_a_lasso <- coef(initial_a_lasso, s = initial_a_lasso$lambda[1], exact = TRUE)[-1,1]

# initial_coef_a_lasso <- coef(initial_a_lasso, initial_a_lasso$lambda[1])[-1,]
weights_a_lasso <- 1 / (abs(initial_coef_a_lasso) + epsilon_a_lasso)

# Step 3: Fit the weighted Lasso model (Adaptive Lasso)
adaptive_lasso <- glmnet(X_matrix, Y, alpha = 1, penalty.factor = weights_a_lasso)

# extract the best lambda value
a_lasso_best_lambda <- l_cv_best_lambda
cat("Adaptive Lasso Best lambda: ", a_lasso_best_lambda, "\n")

# Need to use coef() with the selected best lambda on cv object
a_lasso_coef <- coef(adaptive_lasso, s = a_lasso_best_lambda)  # Extract coefficients at best lambda
a_lasso_coef <- as.vector(a_lasso_coef)
a_lasso_non_zero_coef <- a_lasso_coef[a_lasso_coef != 0][-1]  #Subtract 1 for the intercept
  # sum(a_lasso_coef != 0) 
cat("Adaptive Lasso Number of non-zero coefficients: ", length(a_lasso_non_zero_coef), "\n")# Subtract 1 for the intercept
  

####################################################################################################################################

# PLUGIN LASSO
# Fit the Lasso model
set.seed(12345)
lasso <- glmnet(X_matrix, Y, alpha = 1)

# Example of setting a plugin lambda (TBD)
lambda_plugin <- 0.05
plugin_lasso_model <- glmnet(X_matrix, Y, alpha = 1, lambda = c(lambda_plugin))


# Extracting lambda
lambda_plugin <- plugin_lasso_model$lambda
cat("Plugin Lasso lambda: ", lambda_plugin, "\n")

# Extracting coefficients
coef_plugin_lasso <- coef(plugin_lasso_model, s = lambda_plugin)
coef_plugin_lasso <- as.vector(coef_plugin_lasso)
plugin_lasso_non_zero_coef <- (coef_plugin_lasso[coef_plugin_lasso != 0])[-1]  #Subtract 1 for the intercept
cat("Plugin Lasso Number of non-zero coefficients: ", length(plugin_lasso_non_zero_coef), "\n")

####################################################################################################################################

lcvcoefs <- coef(lasso_CV)
lcvcoefs_vector <- as.vector(lcvcoefs)
lcvcoefs_names <- rownames(lcvcoefs)
lcvnamed_coefs <- setNames(lcvcoefs_vector, lcvcoefs_names)
lcvnon_zero_coefs <- lcvnamed_coefs[lcvnamed_coefs != 0]
lcvnon_zero_coefs <- lcvnon_zero_coefs[-1] # remove the intercept
#lcvnon_zero_coefs


ladcoefs <- coef(adaptive_lasso, s = a_lasso_best_lambda)
ladcoefs_vector <- as.vector(ladcoefs)
ladcoefs_names <- rownames(ladcoefs)
ladnamed_coefs <- setNames(ladcoefs_vector, ladcoefs_names)
ladnon_zero_coefs <- ladnamed_coefs[ladnamed_coefs != 0]
ladnon_zero_coefs <- ladnon_zero_coefs[-1] # remove the intercept
#ladnon_zero_coefs

lpgcoefs <- coef(plugin_lasso_model, s = lambda_plugin)
lpgcoefs_vector <- as.vector(lpgcoefs)
lpgcoefs_names <- rownames(lpgcoefs)
lpgnamed_coefs <- setNames(lpgcoefs_vector, lpgcoefs_names)
lpgnon_zero_coefs <- lpgnamed_coefs[lpgnamed_coefs != 0]
lpgnon_zero_coefs <- lpgnon_zero_coefs[-1] # remove the intercept
#lpgnon_zero_coefs


##################################### Lasso MODEL COMPARISION ##############################
all_predictor_names <- unique(c(names(lcvnon_zero_coefs), names(ladnon_zero_coefs), names(lpgnon_zero_coefs)))

# Initialize the data frame
all_non_zero_coefs <- data.frame(Predictor = all_predictor_names, Lasso_CV = NA, Lasso_Adaptive = NA, Lasso_Plugin = NA, stringsAsFactors = FALSE)

# Fill the data frame
all_non_zero_coefs$Lasso_CV[all_non_zero_coefs$Predictor %in% names(lcvnon_zero_coefs)] <- lcvnon_zero_coefs[match(all_non_zero_coefs$Predictor[all_non_zero_coefs$Predictor %in% names(lcvnon_zero_coefs)], names(lcvnon_zero_coefs))]
all_non_zero_coefs$Lasso_Adaptive[all_non_zero_coefs$Predictor %in% names(ladnon_zero_coefs)] <- ladnon_zero_coefs[match(all_non_zero_coefs$Predictor[all_non_zero_coefs$Predictor %in% names(ladnon_zero_coefs)], names(ladnon_zero_coefs))]
all_non_zero_coefs$Lasso_Plugin[all_non_zero_coefs$Predictor %in% names(lpgnon_zero_coefs)] <- lpgnon_zero_coefs[match(all_non_zero_coefs$Predictor[all_non_zero_coefs$Predictor %in% names(lpgnon_zero_coefs)], names(lpgnon_zero_coefs))]

# show in decimal instead of scientific notation
options(scipen=999)
all_non_zero_coefs


#######################################################################################################################
# Ridge
set.seed(12345)  # For reproducibility
# ridge_model <- glmnet(x = X_matrix, y = y_vector, alpha = 0)

# Optionally, specify a particular lambda value or use cross-validation to find an optimal lambda
cv_ridge <- cv.glmnet(X_matrix, Y, alpha = 0, nfolds = 10)
best_lambda_ridge <- cv_ridge$lambda.min
cat("Ridge Best lambda: ", best_lambda_ridge, "\n")

coef_ridge <- coef(cv_ridge, s = best_lambda_ridge)

################## MODEL PRIDICTION ############################

# models predictions on test set
X_test_matrix <- as.matrix(X_test)
Y_test <- as.vector(y_test)


# LassoCV
lasso_cv_pred <- predict(lasso_CV, newx = X_test_matrix, s = l_cv_best_lambda)

# model results
lasso_cv_results <- data.frame(
  ME = mean(lasso_cv_pred - Y_test),
  RMSE = sqrt(mean((lasso_cv_pred - Y_test)^2)),
  MAE = mean(abs(lasso_cv_pred - Y_test)),
  MAPE = mean(abs((Y_test - lasso_cv_pred) / Y_test) * 100)
)
#lasso_cv_results

lass_adaptive_pred <- predict(adaptive_lasso, newx = X_test_matrix, s = a_lasso_best_lambda)

# model results
lass_adaptive_results <- data.frame(
  ME = mean(lass_adaptive_pred - Y_test),
  RMSE = sqrt(mean((lass_adaptive_pred - Y_test)^2)),
  MAE = mean(abs(lass_adaptive_pred - Y_test)),
  MAPE = mean(abs((Y_test - lass_adaptive_pred) / Y_test) * 100)
)
#lass_adaptive_results

lass_plugin_pred <- predict(plugin_lasso_model, newx = X_test_matrix, s = lambda_plugin)

# model results
lass_plugin_results <- data.frame(
  ME = mean(lass_plugin_pred - Y_test),
  RMSE = sqrt(mean((lass_plugin_pred - Y_test)^2)),
  MAE = mean(abs(lass_plugin_pred - Y_test)),
  MAPE = mean(abs((Y_test - lass_plugin_pred) / Y_test) * 100)
)
#lass_plugin_results

# Ridge
ridge_pred <- predict(cv_ridge, newx = X_test_matrix, s = best_lambda_ridge)

# model results
ridge_results <- data.frame(
  ME = mean(ridge_pred - Y_test),
  RMSE = sqrt(mean((ridge_pred - Y_test)^2)),
  MAE = mean(abs(ridge_pred - Y_test)),
  MAPE = mean(abs((Y_test - ridge_pred) / Y_test) * 100)
)
# ridge_results

# Fit AR1 model based on BIC score
ar1_model <- arima(y_train, order=c(1,1,0))

# Print the model summary
#print(summary(ar1_model))

# predict the log return using x_test
# Predict the log return using the AR(1) model
ar1_forecast <- predict(ar1_model, n.ahead = length(y_test))
# convert to df
ar1_accuracy <- as.data.frame(accuracy(ar1_forecast$pred, y_test))
# drop MPE
ar1_accuracy <- ar1_accuracy[, -4]


# Fit AR6 model based on AIC score
ar6_model <- arima(y_train, order=c(6,1,0))

# Print the model summary
# print(summary(ar6_model))

# predict the log return using x_test
# Predict the log return using the AR(1) model
ar6_forecast <- predict(ar6_model, n.ahead = length(y_test))

# check the accuracy of the model
# Calculate the accuracy of the AR(1) model
ar6_accuracy <- as.data.frame(accuracy(ar6_forecast$pred, y_test))
# drop MPE
ar6_accuracy <- ar6_accuracy[, -4]


# integrate all results into one data frame
testing_all_results <- rbind(lasso_cv_results, lass_adaptive_results, lass_plugin_results, ridge_results, ar1_accuracy, ar6_accuracy)
rownames(testing_all_results) <- c("Lasso_CV", "Lasso_Adaptive", "Lasso_Plugin", "Ridge", "AR(1)", "AR(6)")
# testing_all_results

# choose the best model based on ME, RMSE, MAE, MAPE
testing_best_model <- rownames(testing_all_results)[which.min(rowSums(testing_all_results))]
# testing_best_model

all_non_zero_coefs
testing_all_results
testing_best_model

#####################################################################################################################
