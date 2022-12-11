# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:40:44 2022

@author: P77111100
"""

# import the required libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt 
import os

fp = 'C:/Users/wilym/Dropbox/6.Learning 學習相關/成功大學 資工所 AI人工智慧碩士在職專班/\
3.修課課程/1111_P76I400_Machine Learning and Data Science/HW3_Kaggle_futuresales/\
competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(fp + 'sales_train.csv')
test = pd.read_csv(fp + 'test.csv')
sample_submission = pd.read_csv(fp + 'sample_submission.csv')
items = pd.read_csv(fp + 'items.csv')
item_categories = pd.read_csv(fp + 'item_categories.csv')
shops = pd.read_csv(fp + 'shops.csv')

#%% 參數說明
# ID - an Id that represents a (Shop, Item) tuple within the test set
# shop_id - unique identifier of a shop
# item_id - unique identifier of a product
# item_category_id - unique identifier of item category
# item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# item_price - current price of an item
# date - date in format dd/mm/yyyy
# date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# item_name - name of item
# shop_name - name of shop
# item_category_name - name of item category

# ID是shop_id和item_id的組合，214200 = 60 * 22170??
# date_block_num是將日期重新編碼由2013年1月~2015年10月, 編為0~33 (one-hot?)
sales_train['month'] = sales_train['date_block_num'].map(lambda x: (x+1)%12)
sales_train['year'] = sales_train['date_block_num'].map(lambda x: (x+1)//12+0)
sales_train['month'] = sales_train['month'].replace([0], 12) #把'0'月置換成'12'月

df_dum = pd.get_dummies(sales_train['month']) #對'month'做one hot encoding
#%%
cat_vs_itemid = sns.displot(data=items, x="item_category_id") #每個category(類別)有幾種item(商品)
shop_vs_itemid = sns.displot(data=test, x="shop_id")
sales = sns.histplot(data=sales_train, x="date_block_num", hue="shop_id",multiple="stack")
sales_month = sns.displot(data=sales_train, x="month")
sales_item = sns.displot(data=sales_train, x="item_id", y='item_cnt_day')
plt.figure(figsize=(16,9))
price_vs_sales = sns.scatterplot(data=sales_train, x="shop_id", y='item_cnt_day', hue="month", size="item_price", sizes = (20, 200))
plt.show()
