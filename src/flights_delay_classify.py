import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
# import warnings
# warnings.filterwarnings('ignore')
import catboost as cb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

rand_seed=1024
target_col="delayed"

def load_data(data_file='../data/flights.csv', target_col=target_col, rand_seed=rand_seed):
    data = pd.read_csv(data_file)
    # contains cols: date,flight_id,route,aircraft_type,scheduled_hour,crew_duty_hours, <br/>
    #                weather_forecast_18h,weather_actual,atc_slot_delay_min,delayed  <br/>
    data = data.sort_values(by="date", ascending=True)
    cnt = len(data)
    data = data.drop_duplicates(keep=False)
    print(f"origin cnt={cnt}, after drop_duplicates cnt={len(data)}")
    train_data = data[data["date"]<'2026-07-03']
    test_data = data[data["date"]>'2026-07-02']
    # data["month"]=data["date"].apply(lambda x: x.split("-")[1])
    data.drop(['date'], axis=1, inplace=True)
    train_data.drop(['date'], axis=1, inplace=True)
    test_data.drop(['date'], axis=1, inplace=True)
    cate_cols = [x for x in data.columns if data[x].dtype not in [np.float32, np.float64] and x != target_col]
    for col in cate_cols:
        data[col] = pd.Categorical(data[col])
    X_train, X_test = train_data.iloc[:, :-1],test_data.iloc[:, :-1]
    y_train, y_test = train_data.iloc[:, -1],test_data.iloc[:, -1]
    # X = data.iloc[:, :-1] 
    # y = data.iloc[:, -1]  
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand_seed, test_size=0.2)
    # X_test1, X_val, y_test1, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_seed)
    # cate_cols_indexs = np.where(X_train.columns.isin(cate_cols))[0]  
    return X_train, X_test, y_train, y_test, cate_cols

def model_evaluation(pre, y_test):
    print('{:-^16}'.format('Accuracy'))
    print(accuracy_score(pre, y_test))
    print('{:-^16}'.format('Recall'))
    print(recall_score(pre, y_test))
    print('{:-^16}'.format('Percision'))
    print(precision_score(pre, y_test))
    print('{:-^18}'.format('F1-score'))
    print(f1_score(pre, y_test))
    print('{:-^18}'.format('Roc_Auc_score'))
    print(roc_auc_score(pre, y_test))


def model_building(X_train, X_test, y_train, y_test, cate_cols):
    X_test1, X_val, y_test1, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_seed)
    cate_cols_indexs = np.where(X_train.columns.isin(cate_cols))[0] 
    clf = cb.CatBoostClassifier(
        iterations=500,
        depth=8,
        boosting_type='Ordered',
        learning_rate=0.1,
        loss_function='Logloss',
        bagging_temperature=0.85,
        od_type='Iter',
        rsm=0.85,
        od_wait=100,
        l2_leaf_reg=3,
        thread_count=8,
        use_best_model=True,
        random_seed=rand_seed
    )
    clf.fit(X_train.values.tolist(),
            y_train.values.tolist(),
            cat_features=cate_cols_indexs,
            verbose_eval=100,
            early_stopping_rounds=100,
            eval_set=(X_val.values.tolist(), y_val.values.tolist()))
    pre = clf.predict(X_test1.values.tolist())
    model_evaluation(pre, y_test1)
    return clf, np.column_stack((y_test1, pre))


def KFold_EM(X_train, X_test, y_train, y_test, cate_cols, clf, n_splits=5, rand_seed=rand_seed):
    cate_cols_indexs = np.where(X_train.columns.isin(cate_cols))[0]  
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rand_seed)
    prob_oof = np.zeros(X_train.shape[0])  
    test_pred_prob = np.zeros(X_test.shape[0])  
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold {}".format(fold_ + 1))
        xdata_trn, ydata_trn = X_train.iloc[trn_idx].values.tolist(), y_train.iloc[trn_idx].values.tolist()
        xdata_val, ydata_val = X_train.iloc[val_idx].values.tolist(), y_train.iloc[val_idx].values.tolist()
        cate_cols_indexs = np.where(X_train.columns.isin(cate_cols))[0]
        clf.fit(xdata_trn,
                ydata_trn,
                cat_features=cate_cols_indexs,
                verbose_eval=100,
                early_stopping_rounds=500,
                eval_set=(xdata_val, ydata_val))
        prob_oof[val_idx] = clf.predict(xdata_val)
        test_pred_prob += clf.predict(X_test.values.tolist()) / n_splits
    model_evaluation(test_pred_prob.astype(int), y_test)

if __name__=="__main__":
    X_train, X_test, y_train, y_test, cate_cols = load_data()
    clf, pre = model_building(X_train, X_test, y_train, y_test, cate_cols)
    clf.save_model("fligts_delay_trained.model")
    np.savetxt("fligts_delay_predicted.txt",pre, fmt="%.2f")
    # KFold_EM(X_train, X_test, y_train, y_test, cate_cols, clf)
    print("end!")

