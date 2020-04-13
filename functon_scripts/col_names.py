''' create a function to name data frames '''
import numpy as np

namedict = {'Pclass': ['1', '2', '3'], 
            'Sex': ['female', 'male'],
            'Cabin': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Z'],
            'Embarked': ['C', 'Q', 'S'],
            'Age': '.',
            'SibSp': '.',
            'Parch': '.',
            'Fare': '.',
            }

def col_names(features=[], extra = []):
    namelist = []
    for feature in features:
        for category in namedict[feature]:
            namelist.append(feature + '_' + category)
    namelist.extend(extra)
    return namelist

def clean_names_(x):
    return x.split('__')[-1]
clean_names = np.vectorize(clean_names_)