# STA314-Alzheimer-s-Disease-Diagnosis-Project

knitr::opts_chunk$set(echo = TRUE)
library(dplyr)
library(randomForest)
library(caret)
data <- read.csv("train.csv")
install.packages("xgboost")
install.packages("caret")
library(xgboost)
library(caret)

# Data Processing
str(data)
#Pre-processing. Make sure categorical and numerical variables are well classified
train$Diagnosis <- as.factor(train$Diagnosis)
train$Gender <- as.factor(train$Gender)
train$Ethnicity <- as.factor(train$Ethnicity)
train$EducationLevel <- as.factor(train$EducationLevel)
train$Smoking <- as.factor(train$Smoking)
train$FamilyHistoryAlzheimers <- as.factor(train$FamilyHistoryAlzheimers)
train$CardiovascularDisease <- as.factor(train$CardiovascularDisease)
train$Diabetes <- as.factor(train$Diabetes)
train$Depression <- as.factor(train$Depression)
train$HeadInjury <- as.factor(train$HeadInjury)
train$Hypertension <- as.factor(train$Hypertension)
train$MemoryComplaints <- as.factor(train$MemoryComplaints)
train$BehavioralProblems <- as.factor(train$BehavioralProblems)
train$Confusion <- as.factor(train$Confusion)
train$Disorientation <- as.factor(train$Disorientation)
train$PersonalityChanges <- as.factor(train$PersonalityChanges)
train$DifficultyCompletingTasks <- as.factor(train$DifficultyCompletingTasks)
train$Forgetfulness <- as.factor(train$Forgetfulness)
train$AlcoholConsumption <- as.numeric(train$AlcoholConsumption)
train$PhysicalActivity <- as.numeric(train$PhysicalActivity)
train$DietQuality <- as.numeric(train$DietQuality)
train$SleepQuality <- as.numeric(train$SleepQuality)
data <- train %>% select(-c(PatientID, DoctorInCharge))

# Split data into predictors (X) and target (y)
X <- data %>% select(-Diagnosis)
y <- data$Diagnosis

#EDA
#Correlation Matrix
install.packages("plotly")
library(plotly)

# Compute the correlation matrix
cor_matrix <- cor(data, use = "complete.obs")
# Create the heatmap
cor_matrix <- cor(data, use = "complete.obs")
fig <- plot_ly(
z = cor_matrix,
type = "heatmap",
colors = colorRamp(c("white", "#FFEBF0", "pink", "deeppink")),
text = round(cor_matrix, 2),
hoverinfo = "text",
showscale = TRUE,
width = 800,
height = 800
) %>%
layout(
title = "Correlation Matrix",
xaxis = list(tickvals = seq_along(colnames(cor_matrix)), ticktext = colnames(cor_matrix)),
yaxis = list(tickvals = seq_along(rownames(cor_matrix)), ticktext = rownames(cor_matrix)),
margin = list(l = 100, r = 100, b = 100, t = 100, pad = 4))
print(fig)

#Visualizing Categorical Variables
library(purrr)
library(gridExtra)
categorical_vars <- c("Gender", "Ethnicity", "EducationLevel", "FamilyHistoryAlzheimers",
"CardiovascularDisease", "Smoking", "Depression", "HeadInjury",
"Hypertension", "MemoryComplaints", "BehavioralProblems",
"Confusion", "Disorientation", "PersonalityChanges",
"DifficultyCompletingTasks", "Forgetfulness")
subset_vars <- categorical_vars[1:4]
plot_categorical_distribution <- function(variable) {
data %>%
group_by(.data[[variable]], Diagnosis) %>%
summarise(count = n(), .groups = "drop") %>%
group_by(.data[[variable]]) %>%
mutate(proportion = count / sum(count)) %>%
ggplot(aes(x = factor(.data[[variable]]), y = count, fill = factor(Diagnosis))) +
geom_bar(stat = "identity", width = 0.7) +
geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
position = position_stack(vjust = 0.5), color = "black", size = 3) +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
labs(title = paste(variable),
x = variable,
y = "Count",
fill = "Diagnosis") +
theme_minimal() +
theme(
plot.title = element_text(size = 10),
axis.title = element_text(size = 8),
axis.text = element_text(size = 7),
legend.position = "bottom",
legend.text = element_text(size = 8))}
plots <- map(subset_vars, plot_categorical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars <- categorical_vars[5:8]
plot_categorical_distribution <- function(variable) {
data %>%
group_by(.data[[variable]], Diagnosis) %>%
summarise(count = n(), .groups = "drop") %>%
group_by(.data[[variable]]) %>%
mutate(proportion = count / sum(count)) %>%
ggplot(aes(x = factor(.data[[variable]]), y = count, fill = factor(Diagnosis))) +
geom_bar(stat = "identity", width = 0.7) +
geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
position = position_stack(vjust = 0.5), color = "black", size = 3) +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
labs(title = paste(variable),
x = variable,
y = "Count",
fill = "Diagnosis") +
theme_minimal() +
theme(
plot.title = element_text(size = 10),
axis.title = element_text(size = 8),
axis.text = element_text(size = 7),
legend.position = "bottom",
legend.text = element_text(size = 8))}
plots <- map(subset_vars, plot_categorical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars <- categorical_vars[9:12]
plot_categorical_distribution <- function(variable) {
data %>%
group_by(.data[[variable]], Diagnosis) %>%
summarise(count = n(), .groups = "drop") %>%
group_by(.data[[variable]]) %>%
mutate(proportion = count / sum(count)) %>%
ggplot(aes(x = factor(.data[[variable]]), y = count, fill = factor(Diagnosis))) +
geom_bar(stat = "identity", width = 0.7) +
geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
position = position_stack(vjust = 0.5), color = "black", size = 3) +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
labs(title = paste(variable),
x = variable,
y = "Count",
fill = "Diagnosis") +
theme_minimal() +
theme(
plot.title = element_text(size = 10),
axis.title = element_text(size = 8),
axis.text = element_text(size = 7),
legend.position = "bottom",
legend.text = element_text(size = 8))}
plots <- map(subset_vars, plot_categorical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars <- categorical_vars[13:16]
plot_categorical_distribution <- function(variable) {
data %>%
group_by(.data[[variable]], Diagnosis) %>%
summarise(count = n(), .groups = "drop") %>%
group_by(.data[[variable]]) %>%
mutate(proportion = count / sum(count)) %>%
ggplot(aes(x = factor(.data[[variable]]), y = count, fill = factor(Diagnosis))) +
geom_bar(stat = "identity", width = 0.7) +
geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
position = position_stack(vjust = 0.5), color = "black", size = 3) +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
labs(title = paste(variable),
x = variable,
y = "Count",
fill = "Diagnosis") +
theme_minimal() +
theme(
plot.title = element_text(size = 10),
axis.title = element_text(size = 8),
axis.text = element_text(size = 7),
legend.position = "bottom",
legend.text = element_text(size = 8))}
plots <- map(subset_vars, plot_categorical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))

#Visualizing Numerical Variables
numerical_vars <- c("Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
"DietQuality", "SleepQuality", "SystolicBP", "DiastolicBP",
"CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
"CholesterolTriglycerides", "MMSE", "FunctionalAssessment",
"ADL")
subset_vars_n <- categorical_vars[1:4]
plot_numerical_distribution <- function(variable) {
# Calculate median values for each Diagnosis
median_values <- data %>%
group_by(Diagnosis) %>%
summarise(median_value = median(.data[[variable]], na.rm = TRUE), .groups = "drop")
data %>%
ggplot(aes(x = factor(Diagnosis), y = .data[[variable]], fill = factor(Diagnosis))) +
geom_boxplot() +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
scale_x_discrete(labels = c("1" = "No Alzheimer's", "2" = "Alzheimer's")) +
labs(title = paste(variable),
x = "Diagnosis",
y = variable,
fill = "Diagnosis") +
theme_minimal() +
geom_text(data = median_values,
aes(x = factor(Diagnosis), y = median_value,
label = round(median_value, 2)),
color = "black", size = 4, vjust = -0.5)}
plots <- map(subset_vars_n, plot_numerical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars_n <- categorical_vars[5:8]
plot_numerical_distribution <- function(variable) {
# Calculate median values for each Diagnosis
median_values <- data %>%
group_by(Diagnosis) %>%
summarise(median_value = median(.data[[variable]], na.rm = TRUE), .groups = "drop")
data %>%
ggplot(aes(x = factor(Diagnosis), y = .data[[variable]], fill = factor(Diagnosis))) +
geom_boxplot() +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
scale_x_discrete(labels = c("1" = "No Alzheimer's", "2" = "Alzheimer's")) +
labs(title = paste(variable),
x = "Diagnosis",
y = variable,
fill = "Diagnosis") +
theme_minimal() +
geom_text(data = median_values,
aes(x = factor(Diagnosis), y = median_value,
label = round(median_value, 2)),
color = "black", size = 4, vjust = -0.5)}
plots <- map(subset_vars_n, plot_numerical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars_n <- categorical_vars[9:12]
plot_numerical_distribution <- function(variable) {
# Calculate median values for each Diagnosis
median_values <- data %>%
group_by(Diagnosis) %>%
summarise(median_value = median(.data[[variable]], na.rm = TRUE), .groups = "drop")
data %>%
ggplot(aes(x = factor(Diagnosis), y = .data[[variable]], fill = factor(Diagnosis))) +
geom_boxplot() +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
scale_x_discrete(labels = c("1" = "No Alzheimer's", "2" = "Alzheimer's")) +
labs(title = paste(variable),
x = "Diagnosis",
y = variable,
fill = "Diagnosis") +
theme_minimal() +
geom_text(data = median_values,
aes(x = factor(Diagnosis), y = median_value,
label = round(median_value, 2)),
color = "black", size = 4, vjust = -0.5)}
plots <- map(subset_vars_n, plot_numerical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))
subset_vars_n <- categorical_vars[13:15]
plot_numerical_distribution <- function(variable) {
# Calculate median values for each Diagnosis
median_values <- data %>%
group_by(Diagnosis) %>%
summarise(median_value = median(.data[[variable]], na.rm = TRUE), .groups = "drop")
data %>%
ggplot(aes(x = factor(Diagnosis), y = .data[[variable]], fill = factor(Diagnosis))) +
geom_boxplot() +
scale_fill_manual(values = c("1" = "grey", "2" = "#FFB6C1"),
labels = c("No Alzheimer's", "Alzheimer's")) +
scale_x_discrete(labels = c("1" = "No Alzheimer's", "2" = "Alzheimer's")) +
labs(title = paste(variable),
x = "Diagnosis",
y = variable,
fill = "Diagnosis") +
theme_minimal() +
geom_text(data = median_values,
aes(x = factor(Diagnosis), y = median_value,
label = round(median_value, 2)),
color = "black", size = 4, vjust = -0.5)}
plots <- map(subset_vars_n, plot_numerical_distribution)
do.call(grid.arrange, c(plots, ncol = 2))

# Best Subset Feature Selection
library(leaps)
full_data <- data.frame(y, X)
# Perform best subset selection
best_subset <- regsubsets(y ~ ., data = full_data, nvmax = ncol(X))
subset_summary <-summary(best_subset)
# Plot metrics to determine optimal number of features
par(mfrow = c(1, 2))
plot(subset_summary$cp, xlab = "Number of Features", ylab = "Cp", type = "b")
plot(subset_summary$adjr2, xlab = "Number of Features", ylab = "Adjusted R²", type = "b")
# Get the number of features with the lowest Cp
optimal_features <- which.min(subset_summary$cp)
print(paste("Optimal number of features:", optimal_features))
optimal_vars <- names(coef(best_subset, optimal_features))
print(optimal_vars)

# 80-20 Train-Test Split
set.seed(135) # For reproducibility
train_index <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#Full random forest model using 32 predictors
rf_model <- randomForest(Diagnosis ~ ., data = train_data, ntree = 500, mtry = 32, importance = TRUE)
rf_model
yhat.bag <- predict(rf_model,test_data)
# Convert both predictions and actual values to numeric
yhat_numeric <- as.numeric(as.character(yhat.bag))
actual_numeric <- as.numeric(as.character(test_data$Diagnosis))
# Calculate Mean Squared Error (MSE)
test_mse <- mean((yhat_numeric - actual_numeric)^2)
print(paste("Test MSE:", test_mse))
yhat.bag <- factor(yhat.bag, levels = c(0, 1))
actual <- factor(test_data$Diagnosis, levels = c(0, 1))
# Evaluate model performance
confusion <- confusionMatrix(yhat.bag, test_data$Diagnosis)
print(confusion)
# Feature importance
importance <- importance(rf_model)
varImpPlot(rf_model)

# Cross-Validation of Predictor Numbers
control <- trainControl(method = "cv", number = 10)
# Define the grid of mtry values
tuneGrid <- expand.grid(mtry = seq(2, ncol(X), by = 2))

# Train the model
rf_cv <- train(
Diagnosis ~ .,
data = train_data,
method = "rf",
metric = "Accuracy",
tuneGrid = tuneGrid,
trControl = control,
ntree = 500)

# Updated random forest model with 16 predictors
rf_model_optimal <- randomForest(Diagnosis ~ ., data = train_data, ntree = 500, mtry = 16, importance =
TRUE)
rf_model_optimal
yhat_optimal <- predict(rf_model_optimal, test_data)
# Convert both predictions and actual values to numeric
yhat_new <- as.numeric(as.character(yhat_optimal))
actual_numeric <- as.numeric(as.character(test_data$Diagnosis))
# Calculate Mean Squared Error (MSE)
test_mse1 <- mean((yhat_new - actual_numeric)^2)
print(paste("Test MSE:", test_mse1))
confusionMatrix(yhat_optimal, test_data$Diagnosis)
print(confusionMatrix)
# Feature importance
importance <- importance(rf_model_optimal)
# Extract variable importance as a data frame
importance_df <- data.frame(
Variable = rownames(importance),
MeanDecreaseAccuracy = importance[, "MeanDecreaseAccuracy"],
MeanDecreaseGini = importance[, "MeanDecreaseGini"])
# Sort
top_features <- importance_df %>%
arrange(desc(MeanDecreaseAccuracy)) %>%
head(7)
# Visualize only the top 7 features
varImpPlot(rf_model_optimal, sort = TRUE, n.var = 7, main = "Top 7 Feature Importance")

# Random Forest Model with five selected features
data1 <- read.csv("train_new.csv")
str(data1)
train_new$MemoryComplaints <- as.factor(train_new$MemoryComplaints)
train_new$BehavioralProblems <- as.factor(train_new$BehavioralProblems)
train_new$Diagnosis <- as.factor(train_new$Diagnosis)
data1 <- train_new %>% select(-c(PatientID, DoctorInCharge))
# Split data into predictors (X) and target (y)
X1 <- data1 %>% select(-Diagnosis)
y1 <- data1$Diagnosis
set.seed(4324)
train_new_index <- createDataPartition(data1$Diagnosis, p = 0.8, list = FALSE)
train_new_data <- data1[train_new_index, ]
test_new_data <- data1[-train_new_index, ]
rf_model1 <- randomForest(Diagnosis ~ ., data = train_new_data, ntree = 500, mtry = 5, importance =
TRUE)
rf_model1
yhat.new <- predict(rf_model1,test_new_data)

# Convert both predictions and actual values to numeric
yhat_numeric1 <- as.numeric(as.character(yhat.new))
actual_numeric1 <- as.numeric(as.character(test_new_data$Diagnosis))
# Calculate Mean Squared Error (MSE)
test_mse1 <- mean((yhat_numeric1 - actual_numeric1)^2)
print(paste("Test MSE:", test_mse1))
yhat.new <- factor(yhat.new, levels = c(0, 1))
actual1 <- factor(test_new_data$Diagnosis, levels = c(0, 1))
# Evaluate model performance
confusion1 <- confusionMatrix(yhat.new, test_new_data$Diagnosis)
print(confusion1)
# Feature importance
importance1 <- importance(rf_model1)
varImpPlot(rf_model1)

#Pre-processing the test data
test$Gender <- as.factor(test$Gender)
test$Ethnicity <- as.factor(test$Ethnicity)
test$EducationLevel <- as.factor(test$EducationLevel)
test$Smoking <- as.factor(test$Smoking)
test$FamilyHistoryAlzheimers <- as.factor(test$FamilyHistoryAlzheimers)
test$CardiovascularDisease <- as.factor(test$CardiovascularDisease)
test$Diabetes <- as.factor(test$Diabetes)
test$Depression <- as.factor(test$Depression)
test$HeadInjury <- as.factor(test$HeadInjury)
test$Hypertension <- as.factor(test$Hypertension)
test$MemoryComplaints <- as.factor(test$MemoryComplaints)
test$BehavioralProblems <- as.factor(test$BehavioralProblems)
test$Confusion <- as.factor(test$Confusion)
test$Disorientation <- as.factor(test$Disorientation)
test$PersonalityChanges <- as.factor(test$PersonalityChanges)
test$DifficultyCompletingTasks <- as.factor(test$DifficultyCompletingTasks)
test$Forgetfulness <- as.factor(test$Forgetfulness)
test$AlcoholConsumption <- as.numeric(test$AlcoholConsumption)
test$PhysicalActivity <- as.numeric(test$PhysicalActivity)
test$DietQuality <- as.numeric(test$DietQuality)
test$SleepQuality <- as.numeric(test$SleepQuality)
test_predictions <- predict(rf_model_optimal, newdata = test)
print(test_predictions)

# Decision Tree Model
library(tree)
tree.AD <- tree(Diagnosis ~ ., data = train_data)
summary(tree.AD)
# Use the fitted tree to predict on the test data
tree_pred <- predict(tree.AD, test_data, type = "class")
# Create a contingency table
table(tree_pred, test_data$Diagnosis)
# Calculate the accuracy
accuracy <- sum(tree_pred == test_data$Diagnosis) / length(tree_pred)
cat("Accuracy: ", accuracy, "\n")
# Plot the tree
plot(tree.AD, cex = 1.2)
text(tree.AD1, pretty = 0, cex = 0.6)

# XG Boosting Model
# For XG boosting, all the variables need to be numeric variable
train$Diagnosis <- as.numeric(train$Diagnosis)
train$Gender <- as.numeric(train$Gender)
train$Ethnicity <- as.numeric(train$Ethnicity)
train$EducationLevel <- as.numeric(train$EducationLevel)
train$Smoking <- as.numeric(train$Smoking)
train$FamilyHistoryAlzheimers <- as.numeric(train$FamilyHistoryAlzheimers)
train$CardiovascularDisease <- as.numeric(train$CardiovascularDisease)
train$Diabetes <- as.numeric(train$Diabetes)
train$Depression <- as.numeric(train$Depression)
train$HeadInjury <- as.numeric(train$HeadInjury)
train$Hypertension <- as.numeric(train$Hypertension)
train$MemoryComplaints <- as.numeric(train$MemoryComplaints)
train$BehavioralProblems <- as.numeric(train$BehavioralProblems)
train$Confusion <- as.numeric(train$Confusion)
train$Disorientation <- as.numeric(train$Disorientation)
train$PersonalityChanges <- as.numeric(train$PersonalityChanges)
train$DifficultyCompletingTasks <- as.numeric(train$DifficultyCompletingTasks)
train$Forgetfulness <- as.numeric(train$Forgetfulness)
train$AlcoholConsumption <- as.numeric(train$AlcoholConsumption)
train$PhysicalActivity <- as.numeric(train$PhysicalActivity)
train$DietQuality <- as.numeric(train$DietQuality)
train$SleepQuality <- as.numeric(train$SleepQuality)
train$Age <- as.numeric(train$Age)
train$DiastolicBP <- as.numeric(train$DiastolicBP)
train$SystolicBP <- as.numeric(train$SystolicBP)
# Remove unnecessary variable
data <- train %>% dplyr::select(-c(PatientID, DoctorInCharge))
# Create train and test group, each 50 percent
train_index <- createDataPartition(data$Diagnosis, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
# Ensure the target variable is numeric variable in both test and train dataset
train_data$Diagnosis <- as.numeric(as.factor(train_data$Diagnosis)) - 1
test_data$Diagnosis <- as.numeric(as.factor(test_data$Diagnosis)) - 1
# Split the data into features (X) and target (y)
X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$Diagnosis
X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$Diagnosis
# Create the DMatrix (XGBoost format)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
# Specify the model's parameters
params <- list(
objective = "binary:logistic", \
eval_metric = "error",
max_depth = 4,
eta = 0.1,
nthread = 2)
# 10-fold cross-validation to choose the best number of trees
cv_result <- xgb.cv(
params = params,
data = dtrain,
nrounds = 100,
nfold = 10,
verbose = 1,
early_stopping_rounds = 10,
maximize = FALSE)
# Train the model using the best number of rounds
xgb_model <- xgboost(
params = params,
data = dtrain,
nrounds = cv_result$best_iteration,
verbose = 1)
# Make predictions on the training data
preds <- predict(xgb_model, X_test)
# Final predictions
final_preds_binary <- ifelse(preds > 0.5, 1, 0)
# Calculate final accuracy
final_accuracy <- mean(final_preds_binary == y_test)
# Confusion Matrix
confusion_matrix <- confusionMatrix(factor(final_preds_binary), factor(y_test))
print(confusion_matrix)
# Ensure test dataset' only contains the predictor columns
data_test <- test %>% dplyr::select(-c(PatientID, DoctorInCharge))
X_test_new <- as.matrix(data_test)
# Make predictions
preds_new <- predict(xgb_model, X_test_new)
