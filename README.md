# 2021 Introduction to Machine Learning Final Project
**系級：生物科技學系**
**學號：0717001**
**姓名：李柏毅**


# Topic : Kinase（RAF1）Compound Inhibitors預測
## 1. Introduction
因為我是生科系的學生，所以想要做跟生科領域研究相關並且結合機器學習的題目，再加上本身有在藥物設計實驗室修專題，所以我就選定這個領域相關的題目，也因為去年暑假我在一間智慧藥物設計公司實習，所以對這方面也有一些基本的認識，希望可以結合看看實習所學跟課堂所學做一個porject。
以下我會先對一些會用到的工具以及比較特別的python package做基本的介紹。
### 題目簡介
* Kinase是什麼？
    也就是所謂的「激酶」，是一種能夠催化從高能分子轉移磷酸肌團到特定受值的酶蛋白，這個過程也叫做磷酸化。而人體內有非常多種激酶，並且這些激酶也與一些病症息息相關。許多病症的產生就是因為激酶的作用影響所造成，常見的人類疾病有：癌症、心血管疾病、自體免疫疾病、發炎性疾病、神經系統疾病等。因此激酶抑制藥物一直都是藥物設計研發領域的熱門方向。
    
* Compound是什麼？
    就如上述所說，許多病症都會牽涉到激酶的作用，而Compound就是俗稱的藥物，也就是一個化學結構，透過藥物跟激酶的結合作用，能有效的抑制激酶產生病症影響，又或者是抑制激酶產生某種化學物質進而引發症狀。
    
* RAF1是什麼？
    RAF1是一種Kinase（激酶），與癌症相關。目前已知的RAF1 kinase inhibitor藥物有：Scorafenib，常被用來驗證新的抑制藥物對RAF1的抑制能力。
    
* RDkit python package
    這是一個計算化學領域非常廣被使用的python開源套件，基於對化合物2D以及3D分子的操作，利用機器學習的方法進行化合物得描述浮生成，其中以fingerprint最為常用，化合物結構相似性計算也很出名。簡而言之，就是一個讓電腦也看得懂化學結構的package。後面我也會再更深入介紹一點fingerprint的原理。

    
### 題目目標
* 現階段：使用者輸入一個想要檢驗的compound，我可以預測該compound是否為RAF1的抑制藥物。
* 未來：這其實可以是一個很大的題目，如果不侷限在只做RAF1的模型，我對每個激酶都建構一個屬於他的預測模型，只要使用者輸入一個compound，我可以直接透過全部的模型預測，跟使用者回報這個藥物有可能是哪些激酶的抑制劑。

### 流程介紹
首先，先對每個想要進行預測的compound做計算化學領域常用的"fingerprint"方法，作為預側模型的features，為了方變位來整和，feature的產出我是用自己額外寫的python檔案產，接著，建構三個模型（decision tree, SVM classifier, XGBoost classifier）用training set 調整參數，再分別用holdout validation 以及 k-fold cross validation進行結果分析。
![](https://i.imgur.com/KqDlAAX.png)


## 2. Data Collection
我使用的資料庫是來自於生科實驗室的現有的資料庫，而這個資料庫的來源是來自於一些比較有權威性的生物化學相關資料庫，但在這些data中也只有smile code，並無法被機器學習所使用，所以在這個project中，feature是使用 rdkit套件中的 morganfingerprint以及MACCS fingerprint產生，所以真正的data input只有compound的結構。而compound的結構用的是SMILES code表示。以下會進一步介紹smile code、fingerprint作為feature的意義以及原理



* SMILES code是什麼？
    中文叫做「簡化分子限性輸入規範」(Simplified molecular input line entry specification)，是一種用ASCII字串明確描述分子結構的規範，他可以有效的表達一個化學結構的樣子，舉例來說，雙鍵就以"="表示、三鍵則用"#"表示、有機物中的碳氮氧磷硫溴氯分別以"C, N, O, P, S, Br, Cl"表示。
    舉例來說：環己烷表示為C1CCCCC1
    ![](https://i.imgur.com/rzBCNTlb.png)

    
* Fingerprint是什麼？
    fingerprint的用意是在表達該化學結構有沒有包含某個比較小的結構分子，如果有的話，那個column(小結構)的值就會是"1"，反之則為"0"。而要怎麼去決定要包含哪些小結構分子作為features，就有以下兩種在計算化學領域比較主流的feature取法。
    * MACCS fingerpint
        這種fingerprint有已經預設好的166種重要的小結構，也就是說凡是取樣的compound化學結構有包含其中的一個小結構，該column就會是"1"。
        舉例來說：
        ![](https://i.imgur.com/sp6WWOt.png)
        左邊的compound只有包含紅送區塊的小結構，所以他只有最後一個column的值是"1"；而右邊的compound除了有含紅色區塊的小結構也包含藍色區塊的小結構，所以在他的feature columns裡面就有兩個features的值是"1"。
    * Morgan fingerprint
        這種fingerprint的取法相較之下就較為複雜了，因為前者的取法是使用其他人供認重要的小結構作為features columns，但是如果我們認為這些小結構或許不會是影響他是否為inhibitors compound的關鍵，就會採用這種fingerprint連同操作。這個fingerprint比較像是random從該結構本身取出不見得重要的小結構。以下圖為例，morgan fingerprint的概念就是取某一個中心原子以不同半徑畫圓，如果畫出來有什麼結構就把它當作該feature column得結構，隨著圓形越畫越大，feature取樣的結構大小也會越來越大，好處是他也許可以抓到一些比較不是研究顯示通常比較重要的小結構，可能不小心抓到一些意外是重要影響力的feature。
![](https://i.imgur.com/kalkPSX.png)

### Feature generation and merge
為了產出features，我有撰寫一隻兩隻程式，一隻是用來產生morgan fingerprint以及MACCS fingerprint，另外一隻只是單純把他們的存成dataframe再concat起來。
1. feature_generate.py
其實這是一隻不小的程式，我在撰寫的時候是想要讓他可以結合我實驗室的其他產feature的工具一起變成一隻完整的程式碼，原本應該是可以產出到六種不同公能的feature，但是這個final project我只選用了裡面比較在市面上有公信力的MACCS以及Morgan fingerprint作為feature，所以可以看到我下的指令裡面有 "-maccs y -ecfp y"，ecfp就是morgan fingerprint的意思。
![](https://i.imgur.com/JRhiD9W.png)
那這隻程式的input檔案就是一個裝有一行一行的compound list文字檔，也就是我上面有提到的藥物，並且是以smile code形式儲存。所以可以看到每一行都會有相對應的compound smile code。
這也就是這次project最直接的data input，只有這一筆一筆的compound smile code是真正的data，其他所謂的feature就都是額外用程式產出來的。
![](https://i.imgur.com/aOGI1bq.png)
執行完檔案以後，就會產出兩個csv檔也就是這次project的主角！一個是maccs.csv一個是ecfp.csv，然後就要用下面的檔案將他們合併！
![](https://i.imgur.com/PgoruhY.png)

2. feature_merge.py
為了將兩種不一樣的fingerprint merge在一起變成可用的input data，我也有另外一隻程式是單純用來merge的，之所以會把那麼簡單的工作寫成程式是因為我希望未來可以建立一個pipeline直接用別的程式執行subprocess把上述兩個步驟包起來就直接完成。所以這兩個程式其實都有支援不只這次final project的兩種feature種類。回歸正題，如圖。
![](https://i.imgur.com/wEC9oVk.png)
可以看到產出完成以後會產生一個經過merge的"merged_features_me.csv"檔案，其中的"me"就是分別代表"maccs"以及"ecfp"。data collection部分也就告一段落。
![](https://i.imgur.com/hWet5Hy.png)
## 3. Preprocessing
### Data input
先把data轉換成dataframe檢視一下data 得內容格式，發現其中的feature都是binary bit形式，所以不需要做任何的轉換，另外可以觀察到feature columns總共有1191個，算是挺多的。
![](https://i.imgur.com/oDwd4kB.png)
另外還要讀取一個label檔案，也就是答案，檔案的來源是我生科實驗室整理的exel檔案，最後一個column有標註"label"，就會是我們本次project的答案。這其實是一個用爬蟲爬下來的檔案，是抓取比較有權威性的生物、化學相關資料庫記錄的inhibitors資料。並且已經有經過空值得處理。
![](https://i.imgur.com/NSNxkDC.png)

### Train Test Split
三種model我使用的train test split都是同一份，方便我去進行比較哪一種model的表現比較好。
我用的是hold out validation (7:3)也就是70%的training set以及30%的testing set。直接使用sklearn套件裡的train_test_split函式切割。
![](https://i.imgur.com/Sdg0Bhv.png)
![](https://i.imgur.com/58QbNuL.png)
![](https://i.imgur.com/6ZSpmOA.png)


## 4. Models
### 模型選擇
* 本學期課程教的：Decision Tree, SVM clasifier
* 自己有興趣選擇的：XGBoost classifier
因為這是一個二分類的題目，所以我選擇比較經典的兩個分類器，決策樹以及SVM作為baseline的概念。另外會選擇XGBoost當作第三個自選的模型是因為XGBoost在Kaggle競賽中很常被拿來做使用，高分榜首也時常都是使用XGBoost作為預測模型，所以我想要趁這個機會接觸看看這個模型厲害的地方在哪。

1. SVM classifier
我直接使用sklearn套件的SVM classifier並且有進行調整參數用gird search。
以下是為了呈現漂亮我的grid search的程式碼（真實程式碼請見ipynb檔案）
得到一組最好的performance是SVC(C=1, gamma=0.01, kernel='poly')

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
跟之前的作業一樣直接使用sklearn的套件。也是有經過grid search調整參數。這比較像是這次project的basline，我想要用一個最基本常用的分類器去評估XGBoost的效果。
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
要另外安裝XGBoost在自己的python環境裡面，這個model是調參調最久的一個。畢竟他是牽涉到gradient的計算，詳細的原理就不在這份報告中贅述了。
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

結果跟我一開始想像的有一點點不一樣，我本來想像XGBoost會完虐剩下兩個，但是準確度上來看，SVM的表現是最好的，再來才是XGBoost，整體來說，三個模型都有一定的判斷力，也說明了fingerprint在化學結構分析當作feature的能力還是很不錯的。
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

我本來想說只做一個validation，但是每次作業都有做k fold cross validation這次不做感覺就怪怪的XD，所以我也做了3-fold cross validation，整體表現比holdout validation又更好了一些些，不過也是合理，因為我的k fold validation是直接對全部的X做，所以裡面也有包含training set，可能多少有一點點overfitting到。但是SVM跟XGBoost的表現還是差不多。
#### confusion matrix comparison
1. SVM classifier
![](https://i.imgur.com/yRPCBpem.png)

2. Decision Tree classifier
![](https://i.imgur.com/9lp6jHwm.png)

3. XGBoost classifier
![](https://i.imgur.com/EMJXkNVm.png)

## 6. Conclusion
從結果可以看出，這個project的可行性還是挺高的，光是使用兩種feature就可以滿有效的判斷是分為kinase inhibitor，但是有幾點還是需要改進的部分是：
1. feature數量太多而且有滿多可能都沒有用的，畢竟在化學結構上的feature就是代表小部分的結構，如果有太多多餘的feature對model不見得是有用的資訊，所以我現在有想到的是用pearson correlation去做相似性的比對，如果兩種feature太過相似，是不是可以直接把該feature刪掉，增加model的效率又不失準確度。
2. 如果未來想要做如同我前面一開始所提及的那種系統（使用者輸入一個陌生的compound結構，我要跟他說最有可能是哪個kinase的inhibitor），我還需要建構至少將近五百個kinase的模型，才有辦法對使用者輸入的compound結構作出預測，現階段只有做到可以判斷是不是RAF1這個kinase的抑制劑，雖然RAF1確實也是挺重要的一個kinase，應該在研究上就有一定的用途了。不過如果未來可以增加其他的kinase inhibitor predict model應該會更加完整。

## 7. What I have learned in this semester
本次報告我做了超級超級久，希望助教喜歡。學期初會選這堂課就是因為去年暑假我在一家智慧藥物設計公司實習，學到了很多機器學習的工具以及觀念，但是感覺都只是會用，比較欠缺一些知識上面的基礎，所以就來修這堂課了。果然老師教了很多觀念以及原理，獲益良多。這堂課的重點我覺得就是助教的用心程度，應該是我修過全部資工課程裡面助教最用心的一堂，有幾乎24小時提供解惑的DISCORD群組，雖然一開始很錯愕居然有助教用這個系統當作討論區XD，但是真的挺有效的讓大家更踴躍的提問以及發言。感謝助教們這學期的辛勞～我也從作業中學到很多機器學習基本的SOP的感覺，包含要怎麼視覺化、評估效能、kaggle競賽初體驗也是印象深刻，雖然只進到前一半的排行哈哈哈。總而言之，是很讚的一堂選修課！

