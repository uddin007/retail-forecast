# Databricks notebook source
# MAGIC %md
# MAGIC ###Install Library

# COMMAND ----------

# MAGIC %pip install prophet

# COMMAND ----------

# dbutils.widgets.text("catalog", "")
# dbutils.widgets.text("schema", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load inputs

# COMMAND ----------

inputCatalog = dbutils.widgets.get("catalog")
inpuSchema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libraries & set log level

# COMMAND ----------

import pandas as pd
import datetime
from prophet import Prophet
import logging

logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load source data
# MAGIC * Provides items sold by store by date

# COMMAND ----------

df = spark.table(f'{inputCatalog}.{inpuSchema}.storesales')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create forecasts by sotres and by items 
# MAGIC * Iterate through each row and identify unused store-item combinations (use each combination only once)
# MAGIC * Convert filtered data to Pandas (this way whole dataset, can contain billions of records, is not converted to Pandas)
# MAGIC * Create input dataset for Prophet, prophet_df
# MAGIC * Calculate `periods` based on latest dataset date and target days of forecastig 
# MAGIC * Run model and create output dataset
# MAGIC * Convert output Pandas dataframe to Spark dataframe
# MAGIC * Overwrite using first dataframe, append subsequent ones, this will ensure efficient update of the final dataset

# COMMAND ----------

data_itr = df.rdd.toLocalIterator()

storeItemList = []

counter = 0
 
for row in data_itr:
    storeItem = row['store'] + '-' + row['item']
    if(storeItem not in storeItemList):    
        storeItemList.append(storeItem)
        print(storeItem, storeItemList)
        if (storeItem in storeItemList):
            sdf = df.filter((df.store == row['store']) & (df.item == row['item']))

            pdf = sdf.select('date','quantity').toPandas()
            pdf['date']=pd.to_datetime(pdf['date'])

            max_data_date = pdf['date'].max()
            target_date = datetime.datetime.now() + datetime.timedelta(7)
            foreperiods = (target_date - max_data_date).days
            print(foreperiods)

            prophet_df = pd.DataFrame()
            prophet_df["ds"] = pd.to_datetime(pdf["date"])
            prophet_df["y"] = pdf["quantity"]
            prophet_df = prophet_df.dropna()

            holidays = pd.DataFrame({"ds": [], "holiday": []})
            prophet_obj = Prophet(holidays=holidays)
            prophet_obj.add_country_holidays(country_name='US')
            prophet_obj.fit(prophet_df)

            prophet_future = prophet_obj.make_future_dataframe(periods=foreperiods, freq = "d")
            prophet_forecast = prophet_obj.predict(prophet_future)

            result_df = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][prophet_forecast['ds']>datetime.datetime.now()]
            result_df['store'] = row['store']
            result_df['item'] = row['item']
            result_df['yhat_lower'] = result_df['yhat_lower'].apply(lambda x : x if x > 0 else 0)
            result_df[['yhat', 'yhat_lower', 'yhat_upper']] = result_df[['yhat', 'yhat_lower', 'yhat_upper']].apply(lambda x: pd.Series.round(x, 0))

            if (counter == 0):
                final_df = spark.createDataFrame(result_df)
                final_df.write.mode('overwrite').saveAsTable(f'{inputCatalog}.{inpuSchema}.storeitemforecast')
            if (counter > 0):
                final_df = spark.createDataFrame(result_df)
                final_df.write.mode("append").saveAsTable(f'{inputCatalog}.{inpuSchema}.storeitemforecast')

            counter += 1

# COMMAND ----------

spark.table(f'{inputCatalog}.{inpuSchema}.storeitemforecast').display()

# COMMAND ----------


