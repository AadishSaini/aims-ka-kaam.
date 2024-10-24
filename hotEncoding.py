import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# get data
X = pd.read_csv('./melb_data.csv')
y = X.Price

# split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# achha wala data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# string wala data
obj_cols = [col for col in X.columns if X[col].dtype == 'object']

# data which can be encoded (does not interfere with valid data)
good_cols = [col for col in obj_cols if set(X_valid[col]).issubset(set(X_train[col]))]

# bacha hua data (ie bad data)
not_encodable_cols = list(set(obj_cols) - set(good_cols))  # Corrected: now a list of column names

# remove the bad data
label_X_train = X_train.drop(not_encodable_cols, axis=1)
label_X_valid = X_valid.drop(not_encodable_cols, axis=1)

# init the encoder
ordinal_encoder = OrdinalEncoder()

# apply ordinal
label_X_train[good_cols] = ordinal_encoder.fit_transform(X_train[good_cols])  # Corrected good_cols spelling
label_X_valid[good_cols] = ordinal_encoder.transform(X_valid[good_cols])


object_nunique = list(map(lambda col: X_train[col].nunique(), obj_cols))
d = list(zip(obj_cols, object_nunique))

# high_c=0
# low_c=0

# for a in d:
# 	if a[1] > 10:
# 		high_c+=1
# 	else:
# 		low_c+=1


low_cardinality_cols = [col for col in obj_cols if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(obj_cols)-set(low_cardinality_cols))


OH_encoder = OneHotEncoder(handle_unknown='ignore')

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(obj_cols, axis=1)
num_X_valid = X_valid.drop(obj_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)


OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print(OH_X_train)
print(OH_X_valid)
