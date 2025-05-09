import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv("../data/bank_data_C.csv")  # Replace 'path_to_your_dataset.csv' with the actual file path

# Define the current date
day = pd.Timestamp.today()

recency = df.groupby(["CustomerID"]).agg({"TransactionDate": lambda x: ((day - x.max()).days+1)})
frequency = df.drop_duplicates(subset="TransactionID").groupby(["CustomerID"])[["TransactionID"]].count()
monetary = df.groupby("CustomerID")[["TransactionAmount (INR)"]].sum()

#create rfm table
RFM_table =pd.concat([recency,frequency,monetary], axis = 1)
RFM_table = RFM_table.rename(columns = {"TransactionDate":"recency", "TransactionID":"frequency","TransactionAmount (INR)":"monetary"})

#calculate the quartile for each column 
quantile = RFM_table[['recency','frequency','monetary']].quantile(q= [0.25,0.5,0.75]).to_dict()

def assign_R_score(x, feature):
    if x <= quantile[feature][0.25]:
        return 4
    elif x <= quantile[feature][0.5]:
        return 3
    elif x <= quantile[feature][0.75]:
        return 2
    else: 
        return 1

def assign_M_score(x, feature):
  if x <= quantile[feature][0.25]:
    return 1
  elif x <= quantile[feature][0.5]:
    return 2
  elif x <= quantile[feature][0.75]:
   return 3
  else:
    return 4

def custom_frequency_score (x):
   if x <= 3:
    return x
   else:
       return 4
   
RFM_table["R_score"] = RFM_table["recency"].apply(lambda x: assign_R_score(x, "recency"))
RFM_table["F_score"] = RFM_table["frequency"].apply(custom_frequency_score)
RFM_table["M_score"] = RFM_table["monetary"].apply(lambda x: assign_M_score(x, "monetary"))

RFM_table["RFM_score"]= RFM_table[["R_score","F_score","M_score"]].sum(axis = 1)
RFM_table["RFM_group"]= RFM_table["R_score"].astype(str)+ RFM_table["F_score"].astype(str)+RFM_table["M_score"].astype(str)

def assign_segments(x):
    if x >= 9:
        return 'Best Customers'
    elif x >= 6:
        return 'Loyal Customers'
    elif x >= 4:
        return 'At Risk'
    else:
        return 'Churned'
    
RFM_table["segments"] = RFM_table["RFM_score"].apply(lambda x: assign_segments(x))
RFM_table['weighted_score'] = (RFM_table['R_score'] * 2) + (RFM_table["F_score"] * 1) + (RFM_table['M_score'] * 1)
RFM_table["weighted_segments"] = RFM_table["weighted_score"].apply(lambda x: assign_segments(x))


