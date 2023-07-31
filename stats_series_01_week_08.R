# Snippet 1: Introduction to Logistic Regression
X <- c(1, 2, 3, 4)
y <- factor(c(0, 0, 1, 1)) # Making sure y is a factor
logistic_model <- glm(y ~ X, family = binomial())

# Snippet 2: Building and Interpreting a Logistic Regression Model
probs <- predict(logistic_model, type = "response")
odds_ratios <- exp(coef(logistic_model)) # Calculating Odds Ratios

# Snippet 3: Evaluation Metrics for Classification
library(caret)

predictions <- ifelse(probs > 0.5, 1, 0)
conf_matrix <- confusionMatrix(as.factor(predictions), y)

# Snippet 4: Applications and Real-world Examples (Using previous model)
# Note: This code builds on previous Snippets; apply the model to relevant real-world data
