Prediction of the Manner of the Exercise
===

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## Training
### Pre-processing

```r
library(caret)
setwd("/Users/Nan/Documents/GitRepo/prac_ML/")
training = read.csv("pml-training.csv", stringsAsFactors = F)
dim(training)
```

```
## [1] 19622   160
```


Remove variables with a missing rate >= 0.95.


```r
# names(training)
outcome = training$classe  #save the outcome apart
training = subset(training, select = -classe)

na_rate = apply(training, 2, function(x) sum(is.na(x))/length(x))  #rate of missingness
training = training[, na_rate < 0.95]

(num = sapply(training, is.numeric))  #
```

```
##                       X               user_name    raw_timestamp_part_1 
##                    TRUE                   FALSE                    TRUE 
##    raw_timestamp_part_2          cvtd_timestamp              new_window 
##                    TRUE                   FALSE                   FALSE 
##              num_window               roll_belt              pitch_belt 
##                    TRUE                    TRUE                    TRUE 
##                yaw_belt        total_accel_belt      kurtosis_roll_belt 
##                    TRUE                    TRUE                   FALSE 
##     kurtosis_picth_belt       kurtosis_yaw_belt      skewness_roll_belt 
##                   FALSE                   FALSE                   FALSE 
##    skewness_roll_belt.1       skewness_yaw_belt            max_yaw_belt 
##                   FALSE                   FALSE                   FALSE 
##            min_yaw_belt      amplitude_yaw_belt            gyros_belt_x 
##                   FALSE                   FALSE                    TRUE 
##            gyros_belt_y            gyros_belt_z            accel_belt_x 
##                    TRUE                    TRUE                    TRUE 
##            accel_belt_y            accel_belt_z           magnet_belt_x 
##                    TRUE                    TRUE                    TRUE 
##           magnet_belt_y           magnet_belt_z                roll_arm 
##                    TRUE                    TRUE                    TRUE 
##               pitch_arm                 yaw_arm         total_accel_arm 
##                    TRUE                    TRUE                    TRUE 
##             gyros_arm_x             gyros_arm_y             gyros_arm_z 
##                    TRUE                    TRUE                    TRUE 
##             accel_arm_x             accel_arm_y             accel_arm_z 
##                    TRUE                    TRUE                    TRUE 
##            magnet_arm_x            magnet_arm_y            magnet_arm_z 
##                    TRUE                    TRUE                    TRUE 
##       kurtosis_roll_arm      kurtosis_picth_arm        kurtosis_yaw_arm 
##                   FALSE                   FALSE                   FALSE 
##       skewness_roll_arm      skewness_pitch_arm        skewness_yaw_arm 
##                   FALSE                   FALSE                   FALSE 
##           roll_dumbbell          pitch_dumbbell            yaw_dumbbell 
##                    TRUE                    TRUE                    TRUE 
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell   kurtosis_yaw_dumbbell 
##                   FALSE                   FALSE                   FALSE 
##  skewness_roll_dumbbell skewness_pitch_dumbbell   skewness_yaw_dumbbell 
##                   FALSE                   FALSE                   FALSE 
##        max_yaw_dumbbell        min_yaw_dumbbell  amplitude_yaw_dumbbell 
##                   FALSE                   FALSE                   FALSE 
##    total_accel_dumbbell        gyros_dumbbell_x        gyros_dumbbell_y 
##                    TRUE                    TRUE                    TRUE 
##        gyros_dumbbell_z        accel_dumbbell_x        accel_dumbbell_y 
##                    TRUE                    TRUE                    TRUE 
##        accel_dumbbell_z       magnet_dumbbell_x       magnet_dumbbell_y 
##                    TRUE                    TRUE                    TRUE 
##       magnet_dumbbell_z            roll_forearm           pitch_forearm 
##                    TRUE                    TRUE                    TRUE 
##             yaw_forearm   kurtosis_roll_forearm  kurtosis_picth_forearm 
##                    TRUE                   FALSE                   FALSE 
##    kurtosis_yaw_forearm   skewness_roll_forearm  skewness_pitch_forearm 
##                   FALSE                   FALSE                   FALSE 
##    skewness_yaw_forearm         max_yaw_forearm         min_yaw_forearm 
##                   FALSE                   FALSE                   FALSE 
##   amplitude_yaw_forearm     total_accel_forearm         gyros_forearm_x 
##                   FALSE                    TRUE                    TRUE 
##         gyros_forearm_y         gyros_forearm_z         accel_forearm_x 
##                    TRUE                    TRUE                    TRUE 
##         accel_forearm_y         accel_forearm_z        magnet_forearm_x 
##                    TRUE                    TRUE                    TRUE 
##        magnet_forearm_y        magnet_forearm_z 
##                    TRUE                    TRUE
```


There are non-numeric variables that should be numeric, such as "kurtosis_roll_belt". Re-examining the data I found there are records with "#DIV/0!" (probably generated in Excel, orz), which prevented the variable to be read as numeric. Set those to be missing and convert these variables to be numeric.


```r
tmp = as.data.frame(lapply(training[!num], function(x) gsub("#DIV/0!", "", x)), 
    stringsAsFactors = F)
dim(tmp)
```

```
## [1] 19622    36
```

```r
tmp[, 4:36] = as.data.frame(lapply(tmp[, 4:36], as.numeric))  #convert
training[!num] = tmp
```


Then remove non-numeric variables and the first column which is the sequence number. Remove near-zero variables.


```r
num = sapply(training, is.numeric)
num[1] = FALSE
training = training[num]

nzv = nearZeroVar(training, saveMetrics = T)
training = training[!nzv$nzv]
```


Then standardize the variables and generate pc's.


```r
pre_pro1 = preProcess(training, method = c("center", "scale", "pca"), thresh = 0.9)
trainPC = predict(pre_pro1, training)
```


### Analyze and evaluate

Splice the data into 10-fold subsets. Train and cross validate.


```r
set.seed(123)
folds <- createFolds(outcome, k = 10, list = T, returnTrain = T)

# train(outcome~.,method=)
```



## Testing

```r
testing = read.csv("pml-testing.csv", stringsAsFactors = F)
dim(testing)
```

```
## [1]  20 160
```


Then pre-process the data as what was done for the training set.
