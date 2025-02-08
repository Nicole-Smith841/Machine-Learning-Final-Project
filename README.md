# Machine-Learning-Final-Project
This data set is from kaggle regarding C02 emissions in the United States. The purpose of this linear regression model,
is to predict the C02 emissions in the United States. Initially, after importing the proper functions and libraries, I checked to see if there
were/are any null or missing values using the .isnull() function. After prrprossing the data, I created a chart so I could see a representation of the data 

![ML graph 1](https://github.com/user-attachments/assets/7a6989a3-0162-4948-9bab-d590ba1aca49)

Looking at the data, it would appear that the column containing the totality of the United States is causing the data to skew. Beacuse I want to look at the states 
individually and not as a whole, I decided to drop the rows of data that contained the data of United States. 

After getting the data that I want, the next step is creating a linear regression model of the C02 emissions in the Unitd States. 
#Linear Regression model
X = pd.get_dummies(df[['state-name', 'fuel-name']], drop_first=True)
y = df['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

Once the model has been created and evaluated, the next step is to plot the data and see the regression line. 
Looking at the model, it would seem as though C02 emissions are rising over time and will continue to rise. 
