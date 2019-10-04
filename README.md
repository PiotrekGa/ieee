# Lessons learned

## General
* Spend more time on feature engineering
* Better local validation
* Use script except from Notebook

## Validation
* Use at most 6 folds in Cross-Validation
* Implement `Fast Cross Validation` algorithm in Python
* Use default Catboost as baseline
* Use negative downsampling to create fast new feature validation: [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108616#latest-634925)
* Perform all the 

## Data
* Minify data (function ready)

## Features
* Try engineering all the features: [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/107441)
* Alternatively perform feature selection
* Use Target Encoding or Frequency encoding for categorical features
* Impute NA data with mean and add binary column indicating where it was added
* Remember to impute on training part only in CV
* Remove low frequency categories (experiment with that)
* Remove low / high occuring categories [kernel](https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix)
* Try removing correlated data (Spearman correlation for trees)
* Change pandas `Object` into `Category`

## Training
* Use GPU for gradient boosting (Kaggle / Colab / Cloud)
* Explore Catboost

## Post competition discussions / kernels

### Series of transactions as frauds, not a single one [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111197#latest-640784)
 The 20+ teams don't post solutions usually. But something happened make me feel it is necessary to do this.
I wanna say I don't know if the the shakeup is a coincidence or not. I post my track about this competition to help someone who wanna learn about this competition. People may have different opinions about whole this mess. You cound ask about the solution or code with comments. Plz just dont argue here.

All my track files are in this github:https://github.com/white-bird/kaggle-ieee
if you wanna only run the best model, you should run:
f3deepwalk.py, fdkey.py,
feV307.py, fiyu.py,
model26.ipynb,
model32.ipynb,
https://www.kaggle.com/whitebird/ieee-internal-blend

LB 9500-9520 : I spent most of my time at here while I try to dig the count/mean/std features which didn't work.

LB 9520-9580 : I realized the bad guys stole the cards and make transactions for money, but cards always have some protects, like the biggest amount for one transaction. So they need to have many similar transactions on one card in a lone period or many cards in a short time. That's the keypoint of this competition ----- the series samples make it fraud, not single sample.
We need to find some "keys" to group the data:

1) V307. There are too many V features. Some are int and some are floats. It's not hard to find out that int means the times this card have transactions with same website/seller, and float means the accumulated amount. Obviously, int + cardid may casue misjudge easily. If you have some baseline models, I recommend you the lib eli5 to find which feature is most important, which leads me to the V307. You can find these eda at model14.ipynb. I use the fe_V307.py to process the feature.

2).deviceinfo & id. Different cards have same amt in same addr with same device. is it strange? So I use fd_key.py to extract them.

3). cardid + D. My teammates found this. All people knows the D features minus days mean a lot. We find the D2 and D15 run through the time best by max all the data, while D2 and D15 has the biggest value. fi_yu.py

4). amt + days + addr1. It is simple but easy to misjudge.

LB 9590-9600: So we all know the fraud sample is fraud because its similar samples is fraud. Why not let the infect of fraud more crazy? Making a two-stage models improve 0.001:model26.ipynb + model32.ipynb

LB 9600-9630: This is caused by a bug. I grouped the keys above and get big improvments offline. However, there is only one key, cardid + D, behave badly online. I used 2~3 days to find out that I grouped them with train and test separately. It make improvments online when I grouped the key with all data. It means the key is not working as other keys to make group features but as a embedding key. Then I wrote some rules to process results with kernels. It's easy to understand but make huge boost:https://www.kaggle.com/whitebird/ieee-internal-blend?scriptVersionId=21198581

And there are other small improvments I don't mention. Post here if you have any question while reading/running my code. 