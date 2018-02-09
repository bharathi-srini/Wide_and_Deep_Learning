import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


myfolder = 'F:/Dropbox/WS1718/Recommand System/instacart/input/'


print('loading files ...')

prior = pd.read_csv(myfolder + 'order_products__prior.csv', dtype={'order_id': np.uint32,
           'product_id': np.uint16, 'reordered': np.uint8, 'add_to_cart_order': np.uint8})

train_orders = pd.read_csv(myfolder + 'order_products__train.csv', dtype={'order_id': np.uint32,
           'product_id': np.uint16, 'reordered': np.int8, 'add_to_cart_order': np.uint8 })

orders = pd.read_csv(myfolder + 'orders.csv', dtype={'order_hour_of_day': np.uint8,
           'order_number': np.uint8, 'order_id': np.uint32, 'user_id': np.uint32,
           'order_dow': np.uint8, 'days_since_prior_order': np.float16})

orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.uint8)
orders.days_since_prior_order = orders.days_since_prior_order.fillna(30).astype(np.uint8)

products = pd.read_csv(myfolder + 'products.csv', dtype={'product_id': np.uint16,
            'aisle_id': np.uint8, 'department_id': np.uint8},
             usecols=['product_id', 'aisle_id', 'department_id'])

print('done loading')


print('merge prior and orders and keep train separate ...')

orders_products = orders.merge(prior, how = 'inner', on = 'order_id')
train_orders = train_orders.merge(orders[['user_id','order_id']], left_on = 'order_id', right_on = 'order_id', how = 'inner')

del prior
gc.collect()

print('Creating features I ...')

# sort orders and products to get the rank or the reorder frequency
prdss = orders_products.sort_values(['user_id', 'order_number', 'product_id'], ascending=True)
prdss['product_time'] = prdss.groupby(['user_id', 'product_id']).cumcount()+1

# getting products ordered first and second times to calculate probability later
sub1 = prdss[prdss['product_time'] == 1].groupby('product_id').size().to_frame('prod_first_orders')
sub2 = prdss[prdss['product_time'] == 2].groupby('product_id').size().to_frame('prod_second_orders')
sub1['prod_orders'] = prdss.groupby('product_id')['product_id'].size()
sub1['prod_reorders'] = prdss.groupby('product_id')['reordered'].sum()
sub2 = sub2.reset_index().merge(sub1.reset_index())
sub2['prod_reorder_probability'] = sub2['prod_second_orders']/sub2['prod_first_orders']
sub2['prod_reorder_ratio'] = sub2['prod_reorders']/sub2['prod_orders']
prd = sub2[['product_id', 'prod_orders','prod_reorder_probability', 'prod_reorder_ratio']]

del sub1, sub2, prdss
gc.collect()

print('Creating features II ...')

# extracting prior information (features) by user
users = orders[orders['eval_set'] == 0].groupby(['user_id'])['order_number'].max().to_frame('user_orders')
users['user_period'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].sum()
users['user_mean_days_since_prior'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].mean()

# merging features about users and orders into one dataset
us = orders_products.groupby('user_id').size().to_frame('user_total_products')
us['eq_1'] = orders_products[orders_products['reordered'] == 1].groupby('user_id')['product_id'].size()
us['gt_1'] = orders_products[orders_products['order_number'] > 1].groupby('user_id')['product_id'].size()
us['user_reorder_ratio'] = us['eq_1'] / us['gt_1']
us.drop(['eq_1', 'gt_1'], axis = 1, inplace = True)
us['user_distinct_products'] = orders_products.groupby(['user_id'])['product_id'].nunique()

# the average basket size of the user
users = users.reset_index().merge(us.reset_index())
users['user_average_basket'] = users['user_total_products'] / users['user_orders']

us = orders[orders['eval_set'] != 0]
us = us[['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
users = users.merge(us)

del us
gc.collect()

print('Finalizing features and the main data file  ...')
# merging orders and products and grouping by user and product and calculating features for the user/product combination
data = orders_products.groupby(['user_id', 'product_id']).size().to_frame('up_orders')
data['up_first_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].min()
data['up_last_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].max()
data['up_average_cart_position'] = orders_products.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
data = data.reset_index()

#merging previous data with users
data = data.merge(prd, on = 'product_id')
data = data.merge(users, on = 'user_id')

#user/product combination features about the particular order
data['up_order_rate'] = data['up_orders'] / data['user_orders']
data['up_orders_since_last_order'] = data['user_orders'] - data['up_last_order']
data = data.merge(train_orders[['user_id', 'product_id', 'reordered']],
                  how = 'left', on = ['user_id', 'product_id'])
data = data.merge(products, on = 'product_id')

del orders_products     #, orders, train_orders
gc.collect()

print(' Training and test data for later use in F1 optimization and training  ...')

#save the actual reordered products of the train set in a list format and then delete the original frames
train_orders = train_orders[train_orders['reordered']==1].drop('reordered',axis=1)
orders.set_index('order_id', drop=False, inplace=True)
train1=orders[['order_id','eval_set']].loc[orders['eval_set']==1]
train1['actual'] = train_orders.groupby('order_id').aggregate({'product_id':lambda x: list(x)})
train1['actual']=train1['actual'].fillna('')
n_actual = train1['actual'].apply(lambda x: len(x)).mean()   # this is the average cart size

test1=orders[['order_id','eval_set']].loc[orders['eval_set']==2]
test1['actual']=' '
traintest1=pd.concat([train1,test1])
traintest1.set_index('order_id', drop=False, inplace=True)

del orders, train_orders, train1, test1
gc.collect()

print('setting dtypes for data ...')

#reduce the size by setting data types
data = data.astype(dtype= {'user_id' : np.uint32, 'product_id'  : np.uint16,
            'up_orders'  : np.uint8, 'up_first_order' : np.uint8, 'up_last_order' : np.uint8,
            'up_average_cart_position' : np.uint8, 'prod_orders' : np.uint16,
            'prod_reorder_probability' : np.float16,
            #'prod_reorder_ratio' : np.float16, 'user_orders' : np.uint8,
            'user_period' : np.uint8, 'user_mean_days_since_prior' : np.uint8,
            'user_total_products' : np.uint8, 'user_reorder_ratio' : np.float16,
            'user_distinct_products' : np.uint8, 'user_average_basket' : np.uint8,
            'order_id'  : np.uint32, 'eval_set' : np.uint8,
            'days_since_prior_order' : np.uint8, 'up_order_rate' : np.float16,
            'up_orders_since_last_order':np.uint8,
            'aisle_id': np.uint8, 'department_id': np.uint8})

data['reordered'].fillna(0, inplace=True)  # replace NaN with zeros (not reordered)
data['reordered']=data['reordered'].astype(np.uint8)

gc.collect()

print('Preparing Train and Test sets ...')

# filter by eval_set (train=1, test=2) and dropp the id's columns (not part of training features)
# but keep prod_id and user_id in test

train = data[data['eval_set'] == 1].drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis = 1)
test =  data[data['eval_set'] == 2].drop(['eval_set', 'user_id', 'reordered'], axis = 1)

check =  data.drop(['eval_set', 'user_id', 'reordered'], axis = 1)

del data
gc.collect()

print('preparing X,y')

X_train, X_eval, y_train, y_eval = train_test_split(
    train[train.columns.difference(['reordered'])], train['reordered'], test_size=0.9, random_state=2)
X_train = X_train.drop(['user_reorder_ratio'], axis=1)
X_eval = X_eval.drop(['user_reorder_ratio'], axis=1)


gc.collect()
np.shape(X_train)
model = Sequential()
model.add(Dense(64, input_dim=18, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_eval, y_eval, batch_size=128)
from keras.models import load_model
model.save(myfolder+'model_mlp.h5')
print(score)
