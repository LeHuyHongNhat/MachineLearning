import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTEN


def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)
#
# ros = RandomOverSampler(random_state=0, sampling_strategy={
#     "director_business_unit_loader": 500,
#     "specialist": 500,
#     "managing_director_small_medium_company": 500,
#     "bereichsleiter": 1000
# })
print(y_train.value_counts())
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print("--------------------------------")
# print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("des_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),  # (6458, 7990)
    # ("feature_selector", SelectKBest(chi2, k=300)),
    ("feature_selector", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])
# result =cls.fit_transform(x_train, y_train)
# print(result.shape)
cls.fit(x_train, y_train)
y_predicted = cls.predict(x_test)
print(classification_report(y_test, y_predicted))
# without min_df and max_df. 850k feature
# accuracy                           0.70      1615
#  macro avg       0.49      0.27      0.27      1615
# weighted avg       0.68      0.70      0.65      1615

# with min_df and max_df, without selectKBest.   8000 feature
# accuracy                           0.74      1615
#  macro avg       0.48      0.30      0.31      1615
# weighted avg       0.70      0.74      0.69      1615

# with min_df and max_df, with selectKBest.   800 feature
# accuracy                           0.76      1615
#  macro avg       0.52      0.31      0.31      1615
# weighted avg       0.75      0.76      0.72      1615


# weighted avg       0.76      0.76      0.74      1615   300
