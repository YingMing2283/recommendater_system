mydata <- read.csv("C:/users/user/Desktop/heart.csv")

# EDA
# CHECK STRUCTURE OF THE DATA
str(mydata)

# Convert variables from integers to numeric and replace missing values with 0
mydata[c("Age", "RestingBP","Cholesterol","MaxHR")] <- lapply(mydata[c("Age", "RestingBP","Cholesterol","MaxHR")], function(x) replace(as.numeric(x), is.na(x), 0))

# Create a vector of variable names
var_names <- c("Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "HeartDisease")
var_names2 <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")

# Missing value
missing_value <-(is.na(var_names))
print(missing_value)

#check whether got any weird value or not
# Loop over each variable and compute the minimum and maximum values
for (i in 1:length(var_names)) {
  var_min <- min(mydata[[var_names[i]]])
  var_max <- max(mydata[[var_names[i]]])
  cat(paste("Variable", i, "minimum value:", var_min, "\n"))
  cat(paste("Variable", i, "maximum value:", var_max, "\n"))
}

# Boxplot check outliers
boxplot(mydata[, sapply(mydata, is.numeric)])

# Create a loop to create histograms for each variable
par(mfrow = c(2, 3))  # Set up a 2x3 grid of plots
for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    hist(df[[col]], main = col)
  }
}


plot(mydata$Age, mydata$Cholesterol, main = "Relationship between Age and Cholesterol", xlab = "Age", ylab = "Cholesterol")




