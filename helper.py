import numpy as np
import pandas as pd
import pickle



models_dict = {
    "DecisionTree" : "DecisionTreeClassifierDpth2",
#     "DecisionTreeUnlimitDepth" : "DecisionTreeClassifierDpthUnlimit",
    "LogisticRegression" : "LogisticRegression",
    "RandomForest" : "RandomForestClassifier",
    "XGBoost" : "XGBClassifier"
}

input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']


class Pipeline:
    def __init__(self, train_df):
        self.train_df = train_df
    
    
    def is_weekend(self, tx_datetime):
        weekday = tx_datetime.weekday()
        is_weekend = weekday>=5

        return int(is_weekend)

    
    def is_night(self, tx_datetime):
        tx_hour = tx_datetime.hour
        is_night = tx_hour<=6

        return int(is_night)

    def get_customer_spending_behaviour_features(self, customer_transactions, windows_size_in_days=[1,7,30]):

        customer_transactions=customer_transactions.sort_values('TX_DATETIME')

        customer_transactions.index=customer_transactions.TX_DATETIME

        for window_size in windows_size_in_days:

            SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
            NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()

            AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW

            customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)

        return customer_transactions
    
    
    def get_count_risk_rolling_window(self, terminal_transactions, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"):

        terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')

        terminal_transactions.index=terminal_transactions.TX_DATETIME

        NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
        NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()

        for window_size in windows_size_in_days:

            NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
            NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()

            NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
            NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY

            RISK_WINDOW=NB_FRAUD_WINDOW/NB_TX_WINDOW

            terminal_transactions[feature+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            terminal_transactions[feature+'_RISK_'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)


        return terminal_transactions
    
    
    def pipeline(self, transaction_data):

        train_df = pd.concat([self.train_df,transaction_data], ignore_index=True).sort_values(by=['TX_DATETIME'])
        train_df = train_df.reset_index()#reset indexes


        transaction_data['TX_DURING_WEEKEND']=transaction_data.TX_DATETIME.apply(self.is_weekend)
        transaction_data['TX_DURING_NIGHT']=transaction_data.TX_DATETIME.apply(self.is_night)
        spending_behaviour_customer_0 = self.get_customer_spending_behaviour_features(
            train_df[train_df.CUSTOMER_ID==transaction_data.CUSTOMER_ID[0]])
        tl = ['CUSTOMER_ID_NB_TX_1DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
        for item  in tl:
            transaction_data[item] = list(spending_behaviour_customer_0[np.isnan(spending_behaviour_customer_0.TX_FRAUD)][item])
        terminal_risk0 = self.get_count_risk_rolling_window(train_df[train_df.TERMINAL_ID==transaction_data.TERMINAL_ID[0]])
        tl = ['TERMINAL_ID_NB_TX_1DAY_WINDOW',
           'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
           'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
           'TERMINAL_ID_RISK_30DAY_WINDOW']
        for item in tl:
            transaction_data[item] = list(terminal_risk0[np.isnan(terminal_risk0.TX_FRAUD)][item])
        transaction_data.fillna(0,inplace=True)

        return transaction_data
   


class Loader:
#     def __init__(self):

    def load_train_df(self):
        train_df = pd.read_pickle('./files/data.pkl')

        return train_df

    def load_model(self, classifier_type):
        if classifier_type in models_dict.keys():
            return pickle.load(open(f'./files/{models_dict[classifier_type]}.sav', 'rb'))
        else:
            return "invalid classifier (use 'Loader.get_classifier_types()')"
        
    
    
class Model:
    def __init__(self):
        self.loader = Loader()
        self.train_df = self.loader.load_train_df()
        self.pipeline = Pipeline(self.train_df)
        
    def get_classifier_types(self):
        return list(models_dict.keys())
    
    def predict(self, classifier_type, transaction_data):
        model = self.loader.load_model(classifier_type)
        transaction_data = self.pipeline.pipeline(transaction_data)
        return model.predict_proba(transaction_data[input_features])[:,1][0]
        
    def get_customer_ids(self):
        dff = np.sort(self.train_df['CUSTOMER_ID'].unique())
        return dff.tolist()


    def get_terminal_ids(self):
        dff = np.sort(self.train_df['TERMINAL_ID'].unique())
        return dff.tolist()
        
        