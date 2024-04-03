# Retail Forecast Model
### Business Case
Sales data originated from retail stores from a typical retail chain usually contains billions of records. It can present multiple stores and products sales amounts by number of items sold. Each item from a specific store can be sold at very different rate than that of others. So, to forecast future demands (number of items to be sold per day) we need to consider the historical trend as store-item pair. In this project, we utilize a sample dataset, typical of a retail chain, and developed a forecasting strategy. Following are the highlights of the various stages of the development.

### Time Series Forecasting using Prophet
Facebook open sourced Prophet, a forecasting tool available in Python and R, in Feb 23, 2017. The key chararctistics of Prophet as forecasting tool is described below: 
- hourly, daily, or weekly observations with at least a few months (preferably a year) of history.
- strong seasonalities like day of week and time of year.
- important holidays that occur at irregular intervals but are known in advance. 
- a reasonable number of missing observations or large outliers.
- historical trend changes, for instance due to product launches or logging changes.
- trends that are non-linear growth curves, where a trend hits a natural limit or saturates.

### How Prophet Predicts
Prophet is an additive regression model with four main components:

- Piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
- Yearly seasonal component modeled using Fourier series.
- Weekly seasonal component using dummy variables.
- User-provided list of important holidays.

### Source Data
A sample dataset typical of retail chain that contains three years sales data for two store and two products. The strategy is applicable to handle large volumne of data with multiple store and products. It may require to implement a filter to remove newly added stores or products and products that are sold only infrequently. This is to ensure enough data available to train the model. 

### Adding Holidays and Special Events 
Holidays are added by creating a dataframe for them. It has two columns (holiday and ds) and a row for each occurrence of the holiday. It must include all occurrences of the holiday, both in the past (back as far as the historical data go) and in the future (out as far as the forecast is being made). By using US as a country, following holidays are added:

![image](https://github.com/uddin007/retailforecast/assets/37245809/cb6416fa-4151-4710-b254-4e6a40599600)

### Prediction
Finally, we can run the `predict` method to forecast our data points. The `yhat` column contains the forecasted values. You can also look at the entire DataFrame to see what other values Prophet generates. Sample, output is provided below:

![image](https://github.com/uddin007/retailforecast/assets/37245809/6d5b43fd-126b-4b27-90cd-97bca6672c93)

Following is a presentatio nof predicted number of units that can be sold at futre dates:

![image](https://github.com/uddin007/retailforecast/assets/37245809/c16bb0ee-e0fc-41bb-a73b-e6b32a03c9de)

### Cross validation
Following is the time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values. Parameters are as follows:
- `horizon` forecast horizon
- `initial` length of initial training period
- `period` spacing between cutoff dates

*cuttoff is the separation between the training and the prediction.*

### Performance evaluations
The performance_metrics functionality is used to compute useful statistics of the prediction performance (yhat, yhat_lower, and yhat_upper compared to y), as a function of the distance from the cutoff (how far into the future the prediction was). The statistics computed are mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), mean absolute percent error (MAPE), median absolute percent error (MDAPE) and coverage of the yhat_lower and yhat_upper estimates. These are computed on a rolling window of the predictions in df_cv after sorting by horizon (ds minus cutoff).

### Create forecasts by sotres and by items in production
* Iterate through each row and identify unused store-item combinations (use each combination only once)
* Convert filtered data to Pandas (this way whole dataset, can contain billions of records, is not converted to Pandas)
* Create input dataset for Prophet, prophet_df
* Calculate `periods` based on latest dataset date and target days of forecastig 
* Run model and create output dataset
* Convert output Pandas dataframe to Spark dataframe
* Overwrite using first dataframe, append subsequent ones, this will ensure efficient update of the final dataset





