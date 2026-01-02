import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape",train_df.shape)
print("Test shape",test_df.shape)

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

target = "SalePrice"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

print(X_train.head())
print(y_train.head())
print(X_test.head())    


print (X_train.isnull().sum())
print (X_test.isnull().sum())


X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("Training set size:", X_train_split.shape)
print("Validation set size:", X_val.shape)

print(y_train_split.shape)
print(y_val.shape)


model = LinearRegression()

model.fit(X_train_split, y_train_split)


y_val_pred = model.predict(X_val)
print("Validation Predictions:", y_val_pred[:5])

print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))

final_model = LinearRegression()


final_model.fit(X_train, y_train)

test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})


submission.to_csv("submission.csv", index=False)
