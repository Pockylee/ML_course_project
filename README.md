# 2021 Introduction to Machine Learning Final Project
# Topic : Kinase（RAF1）Compound Inhibitors Prediction Models
## 1. Introduction
Because I am a student of the Department of Biological Sciences, I want to do topics related to the research in the biological field and combined with machine learning. In addition, I have a topic in the drug design laboratory, so I choose topics related to this field. Because I was an intern in a smart drug design company last summer, I also have some basic understanding of this aspect. I hope I can combine what I learned in the internship and what I learned in the classroom to make a project.
Below I will first give a basic introduction to some of the tools that will be used and some special python packages.
### About the topic
* What is a kinase?
Kinase is an enzymatic protein that catalyzes the transfer of phosphomuscles from high-energy molecules to specific receptors, a process also known as phosphorylation. There are many kinds of kinases in the human body, and these kinases are also closely related to some diseases. Many diseases are caused by the action of kinases, and common human diseases are: cancer, cardiovascular disease, autoimmune disease, inflammatory disease, nervous system disease and so on. Therefore, kinase inhibitor drugs have always been a popular direction in the field of drug design and development.
    
* What is a compound?
As mentioned above, many diseases involve the role of kinases, and Compound is commonly known as a drug, that is, a chemical structure. Through the combination of drugs and kinases, it can effectively inhibit the kinase to produce disease effects, or inhibit the kinase. A chemical is produced that triggers symptoms.
    
* What is RAF1
RAF1 is a kinase that is associated with cancer. The currently known RAF1 kinase inhibitor drugs are: Scorafenib, which is often used to verify the ability of new inhibitor drugs to inhibit RAF1.
    
* RDkit python package
This is a python open source suite widely used in the field of computational chemistry. It is based on the manipulation of 2D and 3D molecules of compounds, and uses machine learning methods to describe the formation of compounds. Fingerprint is the most commonly used, and compound structure similarity calculation is also very important. famous. In short, it is a package that allows computers to understand chemical structures. Later, I will introduce the principle of fingerprint in more depth.

    
### Aims
* At this stage: The user enters a compound to be tested, and I can predict whether the compound is an inhibitor of RAF1.
* Future: This can actually be a big problem. If it is not limited to the RAF1 model, I will build a prediction model for each kinase. As long as the user enters a compound, I can directly pass all the The model predicts and reports to users which kinase inhibitors the drug is likely to be.

### Workflow
First of all, first perform the "fingerprint" method commonly used in the field of computational chemistry for each compound that you want to predict, as the features of the pre-side model. In order to integrate the square displacement, the output of the feature is written in python by myself. Then, three models (decision tree, SVM classifier, XGBoost classifier) were constructed and the parameters were adjusted by the training set, and the results were analyzed by holdout validation and k-fold cross validation respectively.
![](https://i.imgur.com/KqDlAAX.png)


## 2. Data Collection
The database I use is the existing database from the biotechnology laboratory, and the source of this database is from some authoritative biochemistry-related databases, but there are only smile codes in these data, and It cannot be used by machine learning, so in this project, the feature is generated using morganfingerprint and MACCS fingerprint in the rdkit suite, so the real data input only has a compound structure. The structure of compound is represented by SMILES code. The following will further introduce the meaning and principle of smile code and fingerprint as features



* What is a SMILES code?
Chinese called "Simplified molecular input line entry specification" (Simplified molecular input line entry specification), is a specification that uses ASCII strings to clearly describe the molecular structure, it can effectively express the appearance of a chemical structure, for example, a double bond is It is represented by "=", the triple bond is represented by "#", and the carbon, nitrogen, phosphorus, sulfur, bromine and chlorine in organic matter are represented by "C, N, O, P, S, Br, Cl" respectively.
    For an example：Cyclohexane is expressed asC1CCCCC1
    ![](https://i.imgur.com/rzBCNTlb.png)

    
* What is a fingerprint
The purpose of fingerprint is to express whether the chemical structure contains a relatively small structure molecule. If so, the value of the column (small structure) will be "1", otherwise it will be "0". And how to decide which small structure molecules to include as features, there are the following two mainstream feature selection methods in the field of computational chemistry.
    * MACCS fingerpint
       This fingerprint has 166 important small structures that have been preset, that is to say, if the sampled compound chemical structure contains a small structure, the column will be "1".
        For an example：
        ![](https://i.imgur.com/sp6WWOt.png)
        The compound on the left only has a small structure containing the red block, so it only has a value of "1" in the last column; and the compound on the right contains a small structure with a blue block in addition to the small structure containing the red block, so In his feature columns, there are two features whose value is "1".
    * Morgan fingerprint
       This fingerprint method is more complicated by comparison, because the former method is to use the small structures that other people admit to be important as the features columns, but if we think that these small structures may not be the key to affecting whether he is an inhibitor compound or not , this fingerprint will be used in conjunction with the operation. This fingerprint is more like random extracting small structures that are not necessarily important from the structure itself. The following figure is an example. The concept of morgan fingerprint is to take a certain central atom and draw a circle with different radii. If there is any structure drawn, take it as the structure of the feature column. As the circle gets bigger and bigger, the feature sampling The size of the structure will also become larger and larger. The advantage is that he may be able to catch some small structures that are not usually more important than research shows, and may accidentally catch some features that are unexpectedly important.
![](https://i.imgur.com/kalkPSX.png)

### Feature generation and merge
In order to generate features, I have written one and two programs, one is used to generate morgan fingerprint and MACCS fingerprint, and the other is just to save them as dataframes and then concat them.
1. feature_generate.py
In fact, this is not a small program. When I wrote it, I wanted it to be combined with other feature-producing tools in my laboratory into a complete code. It should have been able to produce six different types of code. It is a public feature, but for this final project, I only selected MACCS and Morgan fingerprint, which are more credible in the market, as features, so you can see that the command I issued contains "-maccs y -ecfp y", ecfp is morgan The meaning of fingerprint.
![](https://i.imgur.com/JRhiD9W.png)
Then the input file of this program is a compound list text file with line by line, which is the drug I mentioned above, and it is stored in the form of smile code. So you can see that each line will have a corresponding compound smile code.
This is also the most direct data input of this project. Only this compound smile code is the real data, and the other so-called features are generated by additional programs.
![](https://i.imgur.com/aOGI1bq.png)
After the file is executed, two csv files will be produced, which is the protagonist of this project! One is maccs.csv and the other is ecfp.csv, then use the file below to merge them!
![](https://i.imgur.com/PgoruhY.png)

2. feature_merge.py
In order to merge two different fingerprints together into usable input data, I also have another program that is purely used for merge. The reason why such a simple work is written into a program is because I hope that in the future, a pipeline can be created directly. Use another program to execute subprocess to wrap the above two steps and complete it directly. So these two programs actually have two feature types that support not only this final project. Back to the topic, as shown in Fig.
![](https://i.imgur.com/wEC9oVk.png)
It can be seen that after the output is completed, a merged "merged_features_me.csv" file will be generated, where "me" stands for "maccs" and "ecfp" respectively. The data collection part also comes to an end.
![](https://i.imgur.com/hWet5Hy.png)
## 3. Preprocessing
### Data input
First convert the data into a dataframe and check the content format of the data. It is found that the features are in binary bit form, so no conversion is required. In addition, it can be observed that there are a total of 1191 feature columns, which is quite a lot.
![](https://i.imgur.com/oDwd4kB.png)
In addition, a label file, which is the answer, must be read. The source of the file is the exel file organized by my biotechnology laboratory. The last column marked with "label" will be the answer to our project. This is actually a file crawled down by a crawler, and it is a collection of inhibitor data recorded in a relatively authoritative biological and chemical-related database. And there are already empty values processed.
![](https://i.imgur.com/NSNxkDC.png)

### Train Test Split
The train test splits I use for the three models are all the same, which is convenient for me to compare which model performs better.
I use hold out validation (7:3) which is 70% training set and 30% testing set. Directly use the train_test_split function in the sklearn package to cut.
![](https://i.imgur.com/Sdg0Bhv.png)
![](https://i.imgur.com/58QbNuL.png)
![](https://i.imgur.com/6ZSpmOA.png)


## 4. Models
### Choices
* Taught in the ML course：Decision Tree, SVM clasifier
* Due to my own interest：XGBoost classifier
Because this is a two-category topic, I choose to compare the two classic classifiers, decision tree and SVM as the concept of baseline. In addition, XGBoost is selected as the third model of choice because XGBoost is often used in Kaggle competitions, and the top scorers often use XGBoost as a prediction model, so I want to take this opportunity to get in touch and have a look What's so great about this model?

1. SVM classifier
I use the SVM classifier of the sklearn suite directly and have grid search for tuning parameters.
The following is the code to render my grid search beautiful (see the ipynb file for the actual code)
Get a set of best performance is SVC(C=1, gamma=0.01, kernel='poly')

```python=
svm_clf = svm.SVC()
param_grid = {
    'C':[0.1, 1, 10, 100, 1000],
    'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
    'kernel':['linear', 'poly', 'rbf']
}
svm_clf_tuned = grid_search.best_estimator_
```


2. Decision Tree
Use the sklearn suite directly as in the previous homework. There is also a grid search to adjust the parameters. This is more like the basline of this project. I want to use a most basic and commonly used classifier to evaluate the effect of XGBoost.
```python=
dt_clf = DecisionTreeClassifier(random_state=1)
# Decision tree classifier
param_grid = {
    'max_depth':[3,5,7,None],
    'min_samples_split':[2,3],
    'max_features':['auto', 'sqrt', 'log2']
}
# Get best Decision tree classifier
dt_clf_tuned = grid_search.best_estimator_
```

3. XGBoost classifier
To additionally install XGBoost in your own python environment, this model is the one with the longest parameter tuning. After all, he is involved in the calculation of gradient, and the detailed principle will not be repeated in this report.
```python=
# XGBoost classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate= 0.3)
param_grid = {
    'max_depth':[5],
    'n_estimators':[100,500,1000],
    'learning_rate':[0.01, 0.1, 0.3],
    'colsample_bytree':[0.5,0.7,1],
    'subsample':[0.6,0.8,1],
    'eval_metric':['mlogloss'],
    'use_label_encoder':[False]
}
# Get best XGBoost classifier
xgb_clf_tuned = grid_search.best_estimator_

```

## 5. Results

### Holdout validation (7:3)


| Holdout Validation| Accuracy | Precision|Recall|
| -------- | -------- | -------- |---------|
| SVM     | 0.8384615384615385   | 0.8922155688622755  |0.861271676300578|
| Decision Tree | 0.7807692307692308  | 0.8766233766233766  |0.7803468208092486 |
| XGBoost     | 0.8346153846153846   | 0.8869047619047619  |0.861271676300578|

The result is a little different from what I imagined at the beginning. I originally thought that XGBoost would kill the remaining two, but in terms of accuracy, the performance of SVM is the best, and then comes XGBoost. Overall, the three Each model has a certain degree of judgment, which also shows that the ability of fingerprint as a feature in chemical structure analysis is still very good.
#### confusion matrix comparison
1. SVM classifier
![](https://i.imgur.com/A6N761Ym.png)

2. Decision Tree classifier
![](https://i.imgur.com/NlAetSym.png)

3. XGBoost classifier
![](https://i.imgur.com/5SKZoc3m.png)

### K-fold cross validation (K=3)

| K-fold cross validation| Accuracy | Precision|Recall|
| -------- | -------- | -------- |--|
| SVM     | 0.8368055555555556    | 0.8786127167630058 |0.8539325842696629|
| Decision Tree |0.8043981481481481   |  0.8502879078694817    |0.8295880149812733|
| XGBoost     | 0.8368055555555556    | 0.851520572450805  |0.8913857677902621|

I originally wanted to say that I only need to do one validation, but I have to do k fold cross validation every time I do it. XD, it feels weird if I don't do it this time, so I also did 3-fold cross validation, and the overall performance is better than holdout validation. Some, but also reasonable, because my k fold validation is done directly on all X, so it also contains the training set, which may be a little bit overfitting. But the performance of SVM and XGBoost is still similar.
#### confusion matrix comparison
1. SVM classifier
![](https://i.imgur.com/yRPCBpem.png)

2. Decision Tree classifier
![](https://i.imgur.com/9lp6jHwm.png)

3. XGBoost classifier
![](https://i.imgur.com/EMJXkNVm.png)

## 6. Conclusion
It can be seen from the results that the feasibility of this project is still quite high. Just using two features can be fully and effectively judged to be a kinase inhibitor, but there are a few points that still need to be improved:
1. There are too many features and it may be useless if there are too many features. After all, the features in the chemical structure represent a small part of the structure. If there are too many redundant features, it may not be useful information for the model, so I now have What comes to mind is to use pearson correlation to compare the similarity. If the two features are too similar, can the feature be deleted directly to increase the efficiency of the model without losing the accuracy.
2. If I want to build a system like the one I mentioned at the beginning (the user enters an unfamiliar compound structure, I have to tell him which kinase inhibitor is most likely), I still need to build at least nearly Five hundred kinase models can only predict the structure of the compound input by the user. At this stage, it is only possible to determine whether it is an inhibitor of the kinase RAF1. Although RAF1 is indeed an important kinase, it should be researched. There is a certain use. However, if other kinase inhibitor prediction models can be added in the future, it should be more complete.


