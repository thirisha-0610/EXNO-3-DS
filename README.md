## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
```
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/f07e3de8-8049-4214-ad51-e699e3e60532)
```
ORDINAL ENCODER
```
```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
```
```
climate=['Hot','Warm','Cold']
```
```
enc=OrdinalEncoder(categories=[climate])
```
```
enc.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/bc63934c-6ea7-4ddd-9c19-126b11b1fe75)
```
df['ord_2']=enc.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/ccfb2c2c-a31e-4e3f-9b8e-fc85ae1a38ed)

```
dfc=df.copy()
```
```
df['ord_2']=le.fit_transform(df['ord_2'])
```
```
dfc
```
![image](https://github.com/user-attachments/assets/84fa76d7-8974-4879-9990-73f5785a3a50)
```
dfc=df.copy()
```
```
df['sum_2']=le.fit_transform(df['ord_2'])
df
```
![image](https://github.com/user-attachments/assets/a26d052a-908a-43b2-8fac-8b7742e39768)
```
ONEHOT ENCODER
```
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()
```
```
enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/79030ec5-d257-469b-a6d1-68d84997c0c0)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d32a4f75-f594-4a25-b1fe-424430fc0024)
```
pip install --upgrade category_encoders
```
```
BINARY ENCODER
```
```
from category_encoders import BinaryEncoder
```
```
import pandas as pd
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/318e5c95-c824-4f69-87a6-47a9ccd9a39c)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/4dcd6d61-216d-4e60-acb3-25a9431dc70d)
```
TARGET ENCODER
```
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/124cabda-73ae-4948-8e1e-64e6f4a6197f)
```
FEATURE ENGINEERING
```
```
import pandas as pd
import numpy as np
from scipy import stats
```
```
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/338e4ab5-6b97-4dfd-99fe-f08bf0652b29)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/c9f69946-2aea-4b35-a8e3-4e2733cdb033)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7f906246-19d6-4ed9-90eb-e880e3e9b9d1)
```
np.reciprocal(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/8e1d44a5-758a-4c78-aeda-5e138fd1dfc2)
```
np.sqrt(df["Highly Negative Skew"])
```
![image](https://github.com/user-attachments/assets/3931d996-423e-4352-b668-358553ea8bda)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/2fa2af31-98db-4a34-9921-a57f58c9f561)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b03e40a2-34b2-4876-afb9-f031cd3cafe5)
```
df['Highly Positive Skew_boxcox'],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/e746bd0d-a93d-4e7c-b716-d41346b291d9)

```
df['Moderate Negative Skew']=np.reciprocal(df['Moderate Negative Skew'])
df
```
![image](https://github.com/user-attachments/assets/3a48a8eb-0e52-4042-842d-28c03f6cbb6b)
```
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9da4e2a0-2b14-446c-a77b-01a8f8826097)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c391b101-85c5-4c64-af6d-b7a16a258ada)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a981f016-270e-4b90-8365-cfd6d99a45f0)

# RESULT:
       Thus, performing Feature Encoding and Transformation process for the given data set is completed.

       
