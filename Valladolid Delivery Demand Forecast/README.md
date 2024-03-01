**Model Description:**

This model uses a time-series model, as well as a clustering model.

In the current iteration, we have very little historical data available, so choose shorter periods to forecast (1 - 3 days) until there is much more data. If you have less than one month’s data, only forecast for a day. If historical data over several years exists, this programme can be used to forecast a month ahead.

The number of clusters used is to be determined beforehand. In our analysis, we found that three (3) was the ideal number of clusters to use. If the delivery trends change, or if it is to be used by other living labs, the number of clusters to be used will need to be reevaluated.

**Input Files and parameters:**

An excel file containing:

-   Date of delivery
-   Delivery time slot
-   Delivery location (latitude and longitude)
-   Method of delivery
-   Delivery vehicle type

Other parameters include number of clusters the delivery locations are to be split into.

**Outputs:**

For a given future day, the model outputs the forecasted number of parcels going to a particular cluster. The longitudes and latitudes given in the output are not exact locations; they are the locations of the clusters' centroids. We group the delivery locations into clusters and predict the total amount of parcels going to these clusters for a future date.

**Future notes:**

No hyperparameter tuning was done because of the very small amount of data available. In future, when more data is available, it will be possible to add this functionality, as well as improve the model(s) used for forecasting.
