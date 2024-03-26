# Databricks notebook source
# MAGIC %md
# MAGIC ### Time Series Forecasting using Prophet
# MAGIC Facebook open sourced Prophet, a forecasting tool available in Python and R, in Feb 23, 2017. The key chararctistics of Prophet as forecasting tool is described below: 
# MAGIC - hourly, daily, or weekly observations with at least a few months (preferably a year) of history
# MAGIC - strong multiple “human-scale” seasonalities: day of week and time of year
# MAGIC - important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl)
# MAGIC - a reasonable number of missing observations or large outliers
# MAGIC - historical trend changes, for instance due to product launches or logging changes
# MAGIC - trends that are non-linear growth curves, where a trend hits a natural limit or saturates

# COMMAND ----------

# MAGIC %md
# MAGIC ### How Prophet Predicts
# MAGIC Prophet is an additive regression model with four main components:
# MAGIC
# MAGIC - Piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
# MAGIC - Yearly seasonal component modeled using Fourier series.
# MAGIC - Weekly seasonal component using dummy variables.
# MAGIC - User-provided list of important holidays.
# MAGIC
# MAGIC `Prophet fits the model using Stan, and have implemented the core of the Prophet procedure in Stan’s probabilistic programming language. Stan performs the MAP optimization for parameters extremely quickly (<1 second), gives the option to estimate parameter uncertainty using the Hamiltonian Monte Carlo algorithm`

# COMMAND ----------

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

spark_df = spark.table(f'{inputCatalog}.{inpuSchema}.storesales').filter(" store == 'ST-001' and item == 'ITEM-001' and quantity > 10").select('date','quantity')

# COMMAND ----------

spark_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Prophet expects certain column names for its input DataFrame. The date column must be renamed ds, and the column to be forecast should be renamed y

# COMMAND ----------

df = spark_df.toPandas()
prophet_df = pd.DataFrame()
prophet_df["ds"] = pd.to_datetime(df["date"])
prophet_df["y"] = df["quantity"]
prophet_df = prophet_df.dropna()
prophet_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Specify days to forecast as `periods` within `Prophet.make_future_dataframe` 
# MAGIC * Add country holidays

# COMMAND ----------

holidays = pd.DataFrame({"ds": [], "holiday": []})
prophet_obj = Prophet(holidays=holidays)

prophet_obj.add_country_holidays(country_name='US')
prophet_obj.fit(prophet_df)

# COMMAND ----------

prophet_future = prophet_obj.make_future_dataframe(periods=60, freq='d')

# COMMAND ----------

prophet_obj.component_modes

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can run the `predict` method to forecast our data points. The `yhat` column contains the forecasted values. You can also look at the entire DataFrame to see what other values Prophet generates.

# COMMAND ----------

prophet_forecast = prophet_obj.predict(prophet_future)

# COMMAND ----------

prophet_forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at a graph representation of our forecast using `plot`

# COMMAND ----------

prophet_plot = prophet_obj.plot(prophet_forecast)

# COMMAND ----------

prophet_plot2 = prophet_obj.plot_components(prophet_forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use Prophet to identify [changepoints](https://facebook.github.io/prophet/docs/trend_changepoints.html), points where the dataset had an abrupt change. This is especially useful for our dataset because it could identify time periods where Coronavirus cases spiked.

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot

prophet_plot = prophet_obj.plot(prophet_forecast)
changepts = add_changepoints_to_plot(prophet_plot.gca(), prophet_obj, prophet_forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross validation
# MAGIC Following is the time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values. Parameters are as follows:
# MAGIC - `horizon` forecast horizon
# MAGIC - `initial` length of initial training period
# MAGIC - `period` spacing between cutoff dates
# MAGIC
# MAGIC *cuttoff is the separation between the training and the prediction.*

# COMMAND ----------

from prophet.diagnostics import cross_validation
df_cv = cross_validation(prophet_obj, initial='730 days', period='15 days', horizon = '30 days')
df_cv.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance evaluations
# MAGIC The performance_metrics functionality is used to compute useful statistics of the prediction performance (yhat, yhat_lower, and yhat_upper compared to y), as a function of the distance from the cutoff (how far into the future the prediction was). The statistics computed are mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), mean absolute percent error (MAPE), median absolute percent error (MDAPE) and coverage of the yhat_lower and yhat_upper estimates. These are computed on a rolling window of the predictions in df_cv after sorting by horizon (ds minus cutoff)

# COMMAND ----------

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize performance
# MAGIC Cross validation performance metrics is visualized with plot_cross_validation_metric for MAPE. Dots show the absolute percent error for each prediction in df_cv. The blue line shows the MAPE, where the mean is taken over a rolling window of the dots. 

# COMMAND ----------

from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')

# COMMAND ----------


