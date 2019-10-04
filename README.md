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

### Very short summary by winner [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111257#latest-640997)

Still waiting for Chris to make a post about our journey.
Till that time here is a very short summary.

Main magic:

    client identification (uid) using card/D/C/V columns (we found almost all 600 000 unique cards and respective clients)
    uid (unique client ID) generalization by agg to remove train set overfitting for known clients and cards
    categorical features / supportive features for models
    horizontal blend by model / vertical blend by client post-process

Features validation:
We've used several validation schemes to select features:

    Train 2 month / skip 2 / predict 2
    Train 4 / skip 1 / predict 1

We were believing that it was the most stable option to predict future and unknown clients.

Models:
We had 3 main models (with single scores):

    Catboost (0.963915 public / 0.940826 private)
    LGBM (0.961748 / 0.938359)
    XGB (0.960205 / 0.932369)

Simple blend (equal weights) of these models gave us (0.966889 public / 0.944795 private). It was our fallback stable second submission.

The key here is that each model was predicting good a different group of uids in test set:

    Catboost did well on all groups
    XGB - best for known
    LGBM - best for unknown

Predictions:

    6 folds / GroupKfold by month
    almost no fine-tuning for models

Solid post about everything only Chris will be able to prepare.
Please wait for his posts and kernels.

But I'll be glad to answer your questions.

### 11th Place Solution [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111235#latest-641052)

Congrats to all the top teams, especially people at the gold zone. This has been a tough competition to us (at least to me 😎).
Our solution was based on stacking. I personally started this competition with stacking about 20 models and after adding more and more oofs (started with Roman's KFold(k=5, shuffle=False) validation scheme with LB~0.9419),
then I continued by adding oofs from:

    most of the public kernel models,
    LGBM based on data created using categorical features + PCA on numerical features,
    XGB trained trained on different encodings (target, catboost, weight of evidence, … encodings) of categorical features + numerical features,
    Lasso trained on different subsets of numerical features (mean and median imputations),
    Ridge trained on different subsets of numerical features (mean and median imputations),
    LGBM/Ridge trained on different subsets of count features (each feature count encoded),
    LGBM/Ridge trained on count features (each feature count encoded) + onehot encoding of counts,
    LGBM/Ridge trained on count features (each feature count encoded) + onehot encoding of counts (dimension reduced using truncated SVD),
    Gaussian Naive Bayes trained on different subsets of count features (each feature count encoded),
    Multinomial Naive Bayes trained on a subset of count features,
    Univariate Logistic Regression trained on a subset of Vs picked using Spearman correlation between them and the target,
    LGBM/XGB trained on categorical features + some numerical features features (picked using Kolmogorov-Smirnov test),
    a couple of NFFM, FFM models trained on raw categorical features + numerical features using this,
    LibFM trained on a subset of categorical features using this code,
    CatBoost trained on different subsets of categorical + numerical features (all features treated as categorical and fed to CatBoost encoding API),
    NN trained on raw data (embedding of categorical data + different imputation methods for the numerical data) with different architectures,
    LGBM/XGB trained on a subset of 2d, 3d, 4d interactions of categorical features,
    Extra Tree classifier trained on different subsets of PCAed data set (numerical data),
    and KNN trained on a small subset of numerical features.

After this, I ended up with about 900+ oofs. By selecting about 120 oofs out of these oofs using RFE (Recursive Feature Elimination) with LGBM as the classifier, I was able to reach 0.9552 on LB. After that I teamed up with my current teammates where we pooled our features and oofs. ynktk had an XGB trained on about 2000 features that scored 0.9528 and he had created lots of oofs with the same validation method as mine as he worked on his single model. ynktk had been using KFold + brute force feature engineering (various aggregations of the data) from the start and my other teammates were mostly doing FE based on validation using GroupKFold on months so in terms of diversity, it was very helpful. Luckily after a day or two we reached 0.9579.

I started to search for the m*gic (😎) after reading Konstantin's post about making them do friendship and he using CatBoost (to me his magic was most probably about some form of an interaction). First I focused on the interactions of the categorical features especially card1-card6 and addresses then I added Cs and Ds to my interactions. After a lot of experiments with CatBoost I found the feature card1 + date.day - D1 (treated as categorical) and shared it with my teammates (later Youngsoo Lee found out that card1 + card6 + addr1 + addr2 + date.day - D1 was a better option). After focusing on this feature and adding new oofs based on it, we reached AUC=0.9649.

Our best stacked model was a 5 time bagged XGB (it was a bit better than LGBM) trained with about 190 oofs (selected from a pool of about 1100+ oofs) that scores 0.9650 on LB with CV=0.9644. 

### 9th place solution notes [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111234#latest-641105)

I know, 9th place out of 6000+ is a very good place. But well… it doesn't exactly feel so after being 1st on public LB for quite a long time :)
Anyway, this was an interesting competition and - it's better to fly high and fall down, rather than never try flying.

First I would like to congratulate top teams and especially FraudSquad for well deserved 1st place with big gap from others!
Also special congratulations to some people I know better than others - my ex teammates from other competitions @johnpateha and @yryrgogo for getting in gold!

And of course, the biggest thanks to my teammate @kostoglot for great and professional teamwork!
Some key points from our solution

    We heavily used identification of transactions belonging to the same user (as I think all of the top teams did).
    To identify users very helpful feature was ("2017-11-30" + TransactionDT - D1) - this corresponds to some date (like first transaction date, etc) of card - same for all transactions of one card/user. Similar applies to several other D features.
    To check identified users very helpful features were V95, V97, V96 which are previous transaction counts in day, week, month and features V126, V128, V127 - cumulative amount of transactions made by user in previous day, week, month.
    We have different approaches on how to integrate user identification in solution - Konstantin used them as features, I used them in postprocessing and pseudo labelling.
    We used cv setup taking first training data set 3 months for training, 1 month gap removed and 2 last months for validation. This cv setup correlated quite good with public LB, but as it turns out, in some cases it lied to us regarding private set and at some points we moved in a bit wrong direction resulting in falling down to 9th place.
    Models used were LightGBM, Catboost and XGBoost.

Good luck!

### 6th place solution: The Zoo [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111247#latest-641113)

Thanks a lot to Kaggle and IEEE to hosting this competition, I hope that we will have another tabular competition soon enough. Big shout out goes to all my teammates @dott1718, @christofhenkel, @ebouteillon, and @maxjeblick it was really a pleasure to working with them.

From reading the solutions that have already been posted and from expecting what others did, I am pretty sure that most did (similar as in Santander) a very similar thing with some fine-grained tricks. Our solution has maybe 2-3 tricks which might be different to what others did. I am a bit tired, so sorry when not all things are super precise :)

User IDs

As everyone should know by now, identifying users is the key to achieving a good score. We did a lot of trial and error here, and @ebouteillon and @dott1718 were monumental in our identification of users. When we merged we combined our two approaches and in the end @dott1718 crafted a very elegant and well-working solution for user identification. You can find the script of how to do it here (let me call those ids uid6 from now on): https://www.kaggle.com/dott1718/ieee-userid-proxy

Our models used different variations of the user id, with some simpler forms that have much less unique values ranging up to the ids posted before, which are very restrictive but also the most precise ones.

However, now comes the bit more trick part of choosing the CV setup. There are different ways of doing it, and there are different types of leaks you then introduce in your CV setup. For example, one way is to do monthly time-split leak which is also what we chose to use in the end as imho this CV kind-of represents test data the closest. We chose to look mostly at average CV across folds. The important fact here is though that you have ID overlaps between folds and you have more overlaps the more folds you have around your validation fold. IDs overlap way more in closer time proximity. So for example, Fold0 (Christmas) and the last fold perform worse than those folds in the middle, because they have more IDs overlapping to the training folds.

What is then also important is to understand how this overlap will behave in test:

Here you see different IDs (let’s focus on UID6) and on the x-axis the different months of test. The y-axis is the percentage of unique IDs in this month that we also observe in train. As the performance of the models is way better on overlapping ids compared to non-overlapping ones, we could expect private LB to have much lower score than public LB. This also means that it is more important on private LB to have a well working model for non-overlapping customers that you have never observed before in training.

As a result of this, we decided to split our efforts into both developing strong models for overlapping IDs and non-overlappping IDs. We created for both our CV setup as well for test two indices which both mark the overlaps, and the non-overlaps. For example, in a validation fold those transactions where we observe the ID in the other folds would be marked as overlap, and all others as non-overlap.

Features

To be honest, this is probably our weak point as we did not spend too much time on this. Maybe we could further improve our results with better features, but I don’t know. We use a lot of count encoded features (also the uids), as well as other aggregated features. We also introduce aggregated features on UID level. Overall this should not be too different compared to some public kernels.

Models

We utilize three types of models: LGB, Catboost, and Neural Networks all fitted on monthly CV setup. All models use pretty much the same / similar features with some differences of course. In Catboost it was necessary to include the IDs as explicit categorical variables to make it work. Neural networks of course needed some feature transformations and other tricks. @christofhenkel and @maxjeblick did most of the work there, so they are probably the best to elaborate on this further. We additionally did a bunch of pseudo-tagged LGB models where we used predictions from only overlapping UIDs (because we are confident with them) as either soft or hard labels. We use them partly in the blend, but hard for me to say if they help much.

Blending

Our blending is now probably the interesting part. First of all, we decided to do hillclimbing blending, mostly using mean or gmean on raw predictions. But instead of optimizing for overall CV, we optimize separately for overlapping and non-overlapping CV. So first, we only optimize the CV for those indices which we have marked as non-overlapping, and then we do the same for the overlapping indices. The cool thing is now that different models come into the mix for those different indices.

LGBs usually work well for both parts, so they are part of both overlapping and non-overlapping blends. Neural networks help significantly for the non-overlapping part, but not so much for the overlapping part because they don’t overfit heavily on the IDs and it is even hard to force them to. Catboost models work well on both parts, but have some issues on the non-overlaps leaking too much which doesn’t work well on test, so they are only part of overlaps. To summarize: Overlap Blend consists of LGBs + Catboosts and nonoverlap blend consists of LGBs + NNs.

Now the next tricky thing is how to combine them? One way that surprisingly works well is to just fuse them by raw probability. Actually, most of our final models just do it like that. We had some other fancy methods for fusing, starting from simple mean adjustments, towards a tricky rank-based fusing approach that @Dieter came up with. The idea is roughly to rank all predictions using only the coverage model, take the nocoverage part and rank again, build a mapping of those ranks to overall ranks and apply it to the nocoverage ranks.

Post-processing

What further helped us is the following trick. Take all predictions from a customer based on our UID6, and combine the predictions to a single value so that all transactions of a customer have the same value. This gave both a nice boost on CV as well as on LB. I think this is also something that helped us quite significantly on private LB because it makes the ranking much more precise by grouping customers together. One way is to just take the mean of the predictions, what we mostly did is to take 90% quantile of the predictions as this usually worked best on CV.

Final submission

Our blends had a certain random range on both CV as well as LB and as we could not fully decide which one to pick we made a blend of blends by taking rank-average of a bunch of our top blends we had. This was a good choice and it is nearly our top private LB solution we picked by doing so.

In our second submission we tried to gamble a bit more by focusing even more on the non-overlaps. So we did a different CV setup splitting by UID instead of by month, so we had little to no UID leaking in the CV. Then we set the nonoverlap predictions based on a blend of that CV setup. It only scored a bit lower on private LB.

