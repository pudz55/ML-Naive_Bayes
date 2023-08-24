# Importing Pandas
import pandas as pd

# Ignoring Warnings
import warnings
warnings.filterwarnings("ignore")

# Reading the pre processed file back 
Pre_Processed_Dataset = pd.read_csv('Pre_Processed_Dataset.csv')
Pre_Processed_Dataset = Pre_Processed_Dataset.drop(233921)
Pre_Processed_Dataset.reset_index(inplace = True, drop = True)

# counting words
Instance_List = list(Pre_Processed_Dataset['Polarity'])
print("Review for Polarity 0 ->", Instance_List.count(0))
print("Review for Polarity 1 ->", Instance_List.count(1))
print()

"""
Output ->

Review for Polarity 0 -> 82007
Review for Polarity 0 -> 486403
"""

# importing required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# setting bag size list
List_Size_Of_Bags = [2000, 4000, 6000]

# data size
List_Data_Size = [10000, 20000]

# list to save accuracies
Lis_of_Accuracies = []

# takign size of data 
for Data_Size in List_Data_Size:

    # lopping all bag size
    for Bag_Size in List_Size_Of_Bags:

        # print bag size
        print("Selected Bag Size ->", Bag_Size)
        print("Data size ",Data_Size," X 2 -> ", Data_Size*2, sep="")
        
        # creating segments
        Data_Segment_1 = Pre_Processed_Dataset[Pre_Processed_Dataset['Polarity'] == 0].iloc[:Data_Size,:]
        Data_Segment_2 = Pre_Processed_Dataset[Pre_Processed_Dataset['Polarity'] == 1].iloc[:Data_Size,:]
        
        # makign new dataframe
        New_Instance_dataframe = frames = pd.concat([Data_Segment_1, Data_Segment_2 ])
        New_Instance_dataframe.reset_index(inplace = True, drop = True)
        
        # setting features 
        Features_Collection = New_Instance_dataframe['Review']

        # setting labels 
        Label_Collection = New_Instance_dataframe['Polarity']
        
        # making the word bag
        cv = CountVectorizer(max_features = Bag_Size)
        Instance_Features = cv.fit_transform(Features_Collection).toarray()
        Instance_Features = pd.DataFrame(Instance_Features)
        
        # Splitting the dataset into the Training set and Test set
        features_train, features_test, labels_train, labels_test = train_test_split(Instance_Features, Label_Collection, test_size = 0.10, random_state = 123)
        
        # kneighbor classifer
        classifier = KNeighborsClassifier()
        classifier.fit(features_train, labels_train)
        Score_1 = classifier.score(features_test,labels_test)
        print("Score of KNeighborsClassifier is ",round(Score_1,2))
        
        # logistic regression
        classifier = LogisticRegression()
        classifier.fit(features_train, labels_train)
        Score_2 = classifier.score(features_test,labels_test)
        print("Score of Logistic regression is ",round(Score_2,2))
        
        classifier = RandomForestClassifier()
        classifier.fit(features_train, labels_train)  
        Score_3 = classifier.score(features_test,labels_test)
        print("Score of Random Forest is ",round(Score_3,2))
        
        # end
        Lis_of_Accuracies.append([Data_Size, Bag_Size, Score_1, Score_2, Score_3])
        print()

"""
Output -> 

Selected Bag Size -> 2000
Data size 10000 X 2 -> 20000
Score of KNeighborsClassifier is  0.67
Score of Logistic regression is  0.85
Score of Random Forest is  0.84

Selected Bag Size -> 4000
Data size 10000 X 2 -> 20000
Score of KNeighborsClassifier is  0.68
Score of Logistic regression is  0.86
Score of Random Forest is  0.84

Selected Bag Size -> 6000
Data size 10000 X 2 -> 20000
Score of KNeighborsClassifier is  0.68
Score of Logistic regression is  0.86
Score of Random Forest is  0.85

Selected Bag Size -> 2000
Data size 20000 X 2 -> 40000
Score of KNeighborsClassifier is  0.7
Score of Logistic regression is  0.85
Score of Random Forest is  0.84

Selected Bag Size -> 4000
Data size 20000 X 2 -> 40000
Score of KNeighborsClassifier is  0.7
Score of Logistic regression is  0.85
Score of Random Forest is  0.84

Selected Bag Size -> 6000
Data size 20000 X 2 -> 40000
Score of KNeighborsClassifier is  0.7
Score of Logistic regression is  0.85
Score of Random Forest is  0.85
"""

# converting the results into datafrem
Results_Dataset = pd.DataFrame(Lis_of_Accuracies)

# setting the column names of the 
Results_Dataset.columns = ['Data Size', 'Bag Size', 'KNC', 'LR', 'RFC']

# dumping the results into the file 
Results_Dataset.to_csv('Manipulated Results Data.csv',index=False)


