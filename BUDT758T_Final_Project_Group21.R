#load libraries
library(tidyverse)
library(caret)
library(tree)
library(class)
library(ROCR)
library(tm)
library(text2vec)
library(SnowballC)
library(glmnet)
library(ranger)
library(xgboost)

set.seed(1)


# loading the datasets
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")

# Combining train_x and train_y
# Converting the target variable into factor
# Removing perfect_rating_score from the data
data <- cbind(train_x, train_y) %>%
  mutate(high_booking_rate = as.factor(high_booking_rate)) %>%
  select(-c(perfect_rating_score))

summary(data)


# Data Preprocessing
data_prep <- function(data){
  
  #price - remove NAs and replace with mean
  data$price <- ifelse(is.na(data$price),  mean(data$price, na.rm=TRUE), data$price)
  
  #accommodates - remove NAs and replace with mean
  data$accommodates <- ifelse(is.na(data$accommodates), mean(data$accommodates, na.rm=TRUE), data$accommodates)
  
  #cleaning fee
  data$cleaning_fee <- ifelse(is.na(data$cleaning_fee), 0, data$cleaning_fee)
  
  #maximum_nights- if maximum_nights > 28, replace with 28 
  data$maximum_nights = ifelse(data$maximum_nights>28,28,data$maximum_nights)
  
  #cancellation_policy - super_strict_30 to strict
  data$cancellation_policy <- ifelse(data$cancellation_policy %in% c("no_refunds","super_strict_30"),
                                     "strict",data$cancellation_policy)
  data$cancellation_policy <- as.factor(data$cancellation_policy)
  
  #bedrooms,beds - remove NAs and replace with mean
  data$bedrooms <- ifelse(is.na(data$bedrooms), mean(data$bedrooms, na.rm=TRUE), data$bedrooms)
  data$beds <- ifelse(is.na(data$beds), mean(data$beds, na.rm=TRUE), data$beds)
  
  #bathrooms - remove NAs and replace with median 
  data$bathrooms = ifelse(is.na(data$bathrooms), median(data$bathrooms, na.rm=TRUE), data$bathrooms)
  
  #host total listings count - remove NAs and replace with mean
  data$host_total_listings_count <- ifelse(is.na(data$host_total_listings_count),
                                           mean(data$host_total_listings_count,
                                                na.rm=TRUE), data$host_total_listings_count)
  
  # host_response_time - replacing NAs with "no response" and converting to factor
  data$host_response_time = ifelse(is.na(data$host_response_time),"no response",data$host_response_time)
  data$host_response_time = as.factor(data$host_response_time)
  
  # converting room_type to factor
  data$room_type = as.factor(data$room_type)
  
  # market
  data$market = ifelse(data$market %in% c('New York','Los Angeles'), data$market, 'Other')
  data$market = ifelse(is.na(data$market),"Other",data$market)
  data$market = as.factor(data$market)
  
  # security_deposit remove NAs and replace with mean
  data$security_deposit = ifelse(is.na(data$security_deposit),
                            mean(data$security_deposit, na.rm=TRUE),
                            data$security_deposit)
  
  # host_since - calculate the number of days since they became hosts
  data$host_since = as.Date("2024-04-01") - as.Date(data$host_since)
  data$host_since = ifelse(is.na(data$host_since),
                           mean(data$host_since, na.rm=TRUE),data$host_since)
  
  # first_review - calculate the number of days since the date of first review
  data$first_review = as.numeric(as.Date("2024-04-01") - as.Date(data$first_review))
  
  #create new variables
  
  # price per person
  data$price_per_person <- data$price/data$accommodates
  
  # has_cleaning_fee
  data$has_cleaning_fee <- ifelse(data$cleaning_fee == 0, "NO", "YES")
  
  # bed_category
  data$bed_category <- ifelse(data$bed_type == "Real Bed","bed","other")
  data$bed_category = as.factor(data$bed_category)
  
  # property_category
  data$property_category <- case_when(
    data$property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
    data$property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
    data$property_type %in% c("Townhouse", "Condominium") ~ "condo",
    data$property_type %in% c("Bungalow", "House") ~ "house",
    .default = "other")
  data$property_category = as.factor(data$property_category)
  
  # charges_for_extra
  data$charges_for_extra = ifelse(data$extra_people %in% c(0,is.na(data$extra_people)), "NO", "YES")
  data$charges_for_extra = as.factor(data$charges_for_extra)
  
  # log_price
  data$log_price = log(data$price)
  
  # log_max_nights
  data$log_max_nights = log(data$maximum_nights)
  
  # ppp_ind 
  data <- data %>%
    group_by(property_category) %>%
    mutate(ppp_ind = ifelse(price_per_person > median(price_per_person),1,0)) %>%
    ungroup()
  data$ppp_ind = as.factor(data$ppp_ind)
  
  # host_acceptance
  data$host_acceptance = case_when(
    data$host_acceptance_rate == "100%" ~ "ALL",
    is.na(data$host_acceptance_rate) ~ "MISSING",
    .default = "SOME")
  data$host_acceptance = as.factor(data$host_acceptance)
  
  # host_response
  data$host_response = case_when(
    data$host_response_rate == "100" ~ "ALL",
    is.na(data$host_response_rate) ~ "MISSING",
    .default = "SOME")
  data$host_response = as.factor(data$host_response)
  
  # has_min_nights
  data$has_min_nights = ifelse(data$minimum_nights > 1, "YES", "NO")
  data$has_min_nights = as.factor(data$has_min_nights)
  
  # host_is_superhost
  data$host_is_superhost = ifelse(str_detect(data$features,
                                             "Host Is Superhost") == TRUE,1,0)
  data$host_is_superhost = ifelse(is.na(data$host_is_superhost),0,data$host_is_superhost)
  
  # instant_bookable
  data$instant_bookable = ifelse(str_detect(data$features,
                                            "Instant Bookable") == TRUE,1,0)
  data$instant_bookable = ifelse(is.na(data$instant_bookable),0,data$instant_bookable)
  
  #host_has_profile_pic 
  data$host_has_profile_pic = ifelse(str_detect(data$features, 
                                                "Host Has Profile Pic") == TRUE,1,0)
  data$host_has_profile_pic = ifelse(is.na(data$host_has_profile_pic),0,
                                     data$host_has_profile_pic)
 
  
  #is_location_exact
  data$is_location_exact = ifelse(str_detect(data$features, 
                                             "Is Location Exact") == TRUE,1,0)
  data$is_location_exact = ifelse(is.na(data$is_location_exact),0,
                                  data$is_location_exact)
  
  #host_identity_verified
  data$host_identity_verified = ifelse(str_detect(data$features, 
                                                  "Host Identity Verified") == TRUE,1,0)
  data$host_identity_verified = ifelse(is.na(data$host_identity_verified),0,
                                       data$host_identity_verified)
  
  #num_of_features - count the number of features
  data$num_of_features = ifelse(is.na(data$features),0,
                                str_count(data$features,",") + 1)
  
  #num_of_verif - count the number of verifications of host
  data$num_of_verif = ifelse(is.na(data$host_verifications),0,
                             str_count(data$host_verifications,",") + 1)
  
  #num_amenities - count the number of amenities 
  data$num_amenities = ifelse(is.na(data$amenities),0,
                              str_count(data$amenities,",") + 1)
  
  #has_security_deposit - whether security_deposit is present or not
  data$has_security_deposit = case_when(
    is.na(data$security_deposit) ~ "NO",
    .default = "YES")
  data$has_security_deposit = as.factor(data$has_security_deposit)

  #price_per_night
  data$price_per_night = data$price/data$minimum_nights
  
  #bath_per_bedroom - number of bathrooms per bedroom
  data$bath_per_bedroom = data$bathrooms/(data$bedrooms + 1)
  
  #is_weekly_price - does the listing have weekly price
  data$is_weekly_price = ifelse(is.na(data$weekly_price),0,1)
  
  #is_monthly_price - does the listing have monthly price
  data$is_monthly_price = ifelse(is.na(data$monthly_price),0,1)
  
  #same_nhood - whether host and listing are in the same neighborhood
  data$same_nhood = ifelse(data$neighborhood == data$host_neighbourhood,1,0)
  data$same_nhood = ifelse(is.na(data$same_nhood),"missing",data$same_nhood)
  data$same_nhood = as.factor(data$same_nhood)
  
  #long_stay- if the listing allows long stays >= 28 days
  data$long_stay = ifelse(data$maximum_nights>=28,1,0)
  
  # id - primary key variable
  data <- data %>%
    mutate(id = row_number())
  
  return(data)
}

data_cleaned <- data_prep(data)
summary(data_cleaned)


# Text mining on different text features

cleaning_tokenizer <- function(v) {
  v %>%
    space_tokenizer(sep = ',') 
}

# Text Mining on Amenities

#tokenize
it_train_amenities <- itoken(data_cleaned$amenities, 
                             preprocessor = tolower, #preprocessing by converting to lowercase
                             tokenizer = cleaning_tokenizer, 
                             ids = data_cleaned$id, 
                             progressbar = FALSE)

#learn the vocabulary
vocab_amenities <- create_vocabulary(it_train_amenities, ngram = c(1L, 2L))
vocab_amenities2 <- prune_vocabulary(vocab_amenities, vocab_term_max = 20)

#vectorize
vectorizer_amenities <- vocab_vectorizer(vocab_amenities2)
dtm_train_amenities <- create_dtm(it_train_amenities, vectorizer_amenities)
dim(dtm_train_amenities)

tfidf_amenities <- TfIdf$new()
dtm_train_tfidf_amen <- fit_transform(dtm_train_amenities, tfidf_amenities)

# Representing as a regular dataframe
amenities <- data.frame(as.matrix(dtm_train_tfidf_amen))





# Text Mining on host_verifications

#tokenize
it_train_verif <- itoken(data_cleaned$host_verifications, 
                         preprocessor = tolower, #preprocessing by converting to lowercase
                         tokenizer = cleaning_tokenizer, 
                         ids = data_cleaned$id, 
                         progressbar = FALSE)

#learn the vocabulary
vocab_verif  <- create_vocabulary(it_train_verif, ngram = c(1L,1L))
vocab_verif2 <- prune_vocabulary(vocab_verif, vocab_term_max = 10)
#vectorize
vectorizer_verif <- vocab_vectorizer(vocab_verif2)
dtm_train_verif <- create_dtm(it_train_verif, vectorizer_verif)
dim(dtm_train_verif)

tfidf_verif <- TfIdf$new()
dtm_train_tfidf_verif <- fit_transform(dtm_train_verif, tfidf_verif)

# Represented as a regular dataframe
verification <- data.frame(as.matrix(dtm_train_tfidf_verif))


# Text mining on house_rules

cleaning_tokenizer2 <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer 
}


it_train_rules <- itoken(data_cleaned$house_rules, 
                   preprocessor = tolower, #preprocessing by converting to lowercase
                   tokenizer = cleaning_tokenizer2, 
                   ids = data_cleaned$id, 
                   progressbar = FALSE)

#learn the vocabulary
vocab_rules <- create_vocabulary(it_train_rules, ngram = c(1L, 2L))
vocab_rules2 <- prune_vocabulary(vocab_rules, vocab_term_max = 20)

#vectorize
vectorizer_rules <- vocab_vectorizer(vocab_rules2)
dtm_train_rules <- create_dtm(it_train_rules, vectorizer_rules)
dim(dtm_train_rules)

tfidf_rules <- TfIdf$new()
dtm_train_tfidf_rules <- fit_transform(dtm_train_rules, tfidf_rules)

# represented as a regular dataframe
rules <- data.frame(as.matrix(dtm_train_tfidf_rules))

rules <- rules %>%
 mutate(no_rules = NA.) %>%
 select(-NA.)


# Combining the text features with the data_cleaned dataframe

am_verif <- cbind(amenities,verification)
am_verif_rules <- cbind(am_verif, rules)
data_cleaned2 <- cbind(data_cleaned, am_verif_rules)



# Selecting necessary features for the model
airbnb_features <- data_cleaned2 %>%
  select(accommodates, bedrooms,beds,cancellation_policy,cleaning_fee,
         host_total_listings_count,price,ppp_ind, price_per_person, 
         property_category, bed_category, bathrooms, extra_people, 
         host_acceptance, host_response,host_response_time,availability_30,
         availability_60,availability_90, availability_365,num_of_features,
         minimum_nights,market,host_since,first_review,host_is_superhost,
         instant_bookable, latitude, longitude, guests_included,
         high_booking_rate, wireless.internet : no_rules ) 

# Converting the categorical variables into dummies
dummy <- dummyVars( ~ . , data=airbnb_features, fullRank = TRUE)
airbnb_dummy <- data.frame(predict(dummy, newdata = airbnb_features))
airbnb_dummy <- airbnb_dummy %>%
  mutate(high_booking_rate.YES = as.factor(high_booking_rate.YES))

# Removing unwanted features
airbnb_dummy <- airbnb_dummy %>%
  select(-c(carbon.monoxide.detector,
            essentials,google,jumio,facebook,
            linkedin,keep,parti,home,
            pet,hous,pm,pleas,allow,leav))


# Dividing the data into train and test splits
train_insts <- sample(nrow(airbnb_dummy), .7*nrow(airbnb_dummy))
data_train <- airbnb_dummy[train_insts,]
data_valid <- airbnb_dummy[-train_insts,]

# Keeping 5% of data aside for model checking

check_sample <- sample(nrow(data_valid), .05*nrow(data_valid))

data_check <- data_valid[check_sample,]
data_valid <- data_valid[-check_sample,]



# Data Modeling

#================= 1. Logistic Regression =======================================

logistic_rate <- glm(high_booking_rate.YES~., data = data_train, family = "binomial")
probs_log_rate <- predict(logistic_rate, newdata = data_valid, type = "response")
probs_log_rate <- ifelse(is.na(probs_log_rate), 0, probs_log_rate)
assertthat::assert_that(sum(is.na(probs_log_rate))==0)
pred_log <- prediction(probs_log_rate, data_valid$high_booking_rate.YES)
roc_log <-performance(pred_log, "tpr", "fpr")
auc_score_log <- performance(pred_log, measure = "auc")@y.values[[1]]
auc_score_log


#====================== 2. Decision Tree ===========================================

mycontrol = tree.control(nrow(data_train), mincut = 5, minsize = 10, mindev = 0.0005)

full_tree=tree(high_booking_rate.YES ~., control = mycontrol, data_train)



# Finding the best tree size for Decision Tree

tree_sizes <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                80, 85, 90, 95, 100, 105, 110, 120, 130, 140, 150)
va_auc <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


for (i in 1:length(tree_sizes)){
  pruned_tree=prune.tree(full_tree, best = tree_sizes[i])
  pruned_tree_preds <- predict(pruned_tree,newdata=data_valid)
  pruned_tree_preds <- ifelse(pruned_tree_preds[,2] > 0.5, 1,0)
  pred_dt <- prediction(pruned_tree_preds, data_valid$high_booking_rate.YES)
  roc_dt <-performance(pred_dt, "tpr", "fpr")
  auc_score <- performance(pred_dt, measure = "auc")@y.values[[1]]
  va_auc[i] <- auc_score
  
}

# Fitting curve for decision tree model
plot(tree_sizes, va_auc, col = "green", type = 'o',xlab = "Tree Size",
     ylab = "Validation Auc score", 
     main = "Decision Tree: Validation AUC Score vs Tree Size", lwd = 2)

# Taking the tree with the maximum auc score
max_auc_index <- which.max(va_auc)
best_size_dt <- tree_sizes[max_auc_index]

# Pruning the tree with best tree size
best_tree=prune.tree(full_tree, best = best_size_dt)

# Finding Predictions
best_tree_preds <- predict(best_tree,newdata=data_valid)
best_tree_preds <- ifelse(best_tree_preds[,2] > 0.5, 1,0)

# Calculating the roc value and auc score
pred_best_tree <- prediction(best_tree_preds, data_valid$high_booking_rate.YES)
roc_best_dt <-performance(pred_best_tree, "tpr", "fpr")
auc_best_dt <- performance(pred_best_tree, measure = "auc")@y.values[[1]]
auc_best_dt


#=========================== 3. XGBoost ==================================================

# Taking x_train, y_train, x_valid, y_valid for input into XGboost model
x_train <- data_train %>%
  select(-high_booking_rate.YES)

y_train <- data_train$high_booking_rate.YES

x_valid <- data_valid %>%
  select(-high_booking_rate.YES)

y_valid <- data_valid$high_booking_rate.YES

# Creating DMatrix for XGboost
dtrain = xgb.DMatrix(as.matrix(sapply(x_train, as.numeric)), 
                     label=as.matrix(y_train))



# Grid search cross validation for XGboost

grid_search <- function(){
  
  depth_choose <- c(2,3,5,7,10)
  nrounds_choose <- c(400,500,600,800,1000)
  eta_choose <- c(0.1,0.3,0.5,0.7,1)
  
  # Initialize an empty dataframe to store results
  results <- data.frame(depth = numeric(),
                        rounds = numeric(),
                        eta = numeric(),
                        auc = numeric(),
                        auc_train = numeric(),
                        stringsAsFactors = FALSE) 
  
  #nested loops to tune these three parameters
  print('depth, nrounds, eta, auc_score, auc_train')
  for(i in c(1:length(depth_choose))){
    for(j in c(1:length(nrounds_choose))){
      for(k in c(1:length(eta_choose))){
        thisdepth <- depth_choose[i]
        thisnrounds <- nrounds_choose[j]
        thiseta <- eta_choose[k]
        
        inner_bst <- xgboost(data = dtrain,
                             max.depth = thisdepth,
                             eta = thiseta,
                             nrounds = thisnrounds, 
                             objective = "binary:logistic",
                             scale_pos_weight = 3.9,
                             verbosity = 0, verbose = 0)
        
        #validation auc
        inner_bst_pred <- predict(inner_bst, as.matrix(sapply(x_valid, as.numeric)))
        inner_bst_classifications <- ifelse(inner_bst_pred > 0.5, 1, 0)
        inner_pred_mod <- prediction(inner_bst_classifications, y_valid)
        inner_roc_log <-performance(inner_pred_mod, "tpr", "fpr")
        inner_auc_score <- performance(inner_pred_mod, measure = "auc")@y.values[[1]]
        
        #training auc 
        inner_bst_pred_train <- predict(inner_bst, as.matrix(sapply(x_train, as.numeric)))
        inner_bst_classifications_train <- ifelse(inner_bst_pred_train > 0.5, 1, 0)
        inner_pred_mod_train <- prediction(inner_bst_classifications_train, y_train)
        inner_roc_log_train <-performance(inner_pred_mod_train, "tpr", "fpr")
        inner_auc_score_train <- performance(inner_pred_mod_train, measure = "auc")@y.values[[1]]
        
        #print the performance for every combination
        print(paste(thisdepth, thisnrounds, thiseta, inner_auc_score,inner_auc_score_train, sep = ", "))
        # Append the results to the dataframe
        results <- rbind(results, data.frame(depth = thisdepth,
                                             rounds = thisnrounds,
                                             eta = thiseta,
                                             auc = inner_auc_score,
                                             auc_train = inner_auc_score_train))
        
      }
    }
  }
  return(results)
}


va_results <- grid_search()
# finding the combination with highest validation auc
va_results[which.max(va_results$auc),] 

# building model on best combination of parameters
bst <- xgboost(data = dtrain,
               max.depth = 2,
               eta = 0.5,
               nrounds = 600,
               scale_pos_weight = 3.9,
               early_stopping_rounds = 5,
               objective = "binary:logistic")

bst_pred <- predict(bst, as.matrix(sapply(x_valid, as.numeric)))
bst_classifications <- ifelse(bst_pred > 0.5, 1, 0)
pred_mod <- prediction(bst_classifications, y_valid)
roc_xgb <-performance(pred_mod, "tpr", "fpr")
auc_score_xgb <- performance(pred_mod, measure = "auc")@y.values[[1]]
auc_score_xgb

# Fitting curve for XGboost

plot_xgb <- va_results %>%
  filter(rounds == 600 , eta == 0.5)
plot(plot_xgb$depth,plot_xgb$auc,type = 'o',col = "blue", ylim = c(0.6, 1),xlab = "Depth" , ylab="AUC Score", lwd = 2, main = "XGBoost: AUC vs Depth")
lines(plot_xgb$depth, plot_xgb$auc_train ,type = 'o', col = "red", lwd = 2)
legend(x = "bottomright", 
       legend=c("Training auc","Validation auc"),  
       fill = c("red","blue"))


#=============================== 4. Ridge =======================================


grid <- 10^seq(-7,7,length=100)

auc_ridge <- rep(0, length(grid))

for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  
  ridge_mod <- glmnet(as.matrix(sapply(x_train, as.numeric)), as.matrix(y_train), family = "binomial", alpha = 0, lambda = lam)
  
  preds <- predict(ridge_mod, newx = as.matrix(sapply(x_valid, as.numeric)), type = "response")
  pred_ridge <- prediction(preds,y_valid )
  roc_ridge <-performance(pred_ridge, "tpr", "fpr")
  auc_score <- performance(pred_ridge, measure = "auc")@y.values[[1]]
  auc_ridge[i] <- auc_score
}


plot(log10(grid), auc_ridge,ylab = "AUC Score",
     xlab = "Lambda",main = "Ridge: AUC Score vs Lambda")

# getting best-performing lambda
best_validation_index <- which.max(auc_ridge)
best_lambda_ridge <- grid[best_validation_index]

# building model on best performing lambda
best_ridge <- glmnet(as.matrix(sapply(x_train, as.numeric)), as.matrix(y_train), family = "binomial", alpha = 0, lambda = best_lambda_ridge)
preds_ridge <- predict(best_ridge, newx = as.matrix(sapply(x_valid, as.numeric)), type = "response")
best_ridge_prediction <- prediction(preds_ridge, y_valid)
roc_best_ridge <-performance(best_ridge_prediction, "tpr", "fpr")
auc_ridge <- performance(best_ridge_prediction, measure = "auc")@y.values[[1]]
auc_ridge

#============================== 5. Lasso ===========================================

grid <- 10^seq(-7,7,length=100)

auc_lasso <- rep(0, length(grid))

for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  
  lasso_mod <- glmnet(as.matrix(sapply(x_train, as.numeric)), as.matrix(y_train), family = "binomial", alpha = 1, lambda = lam)
  
  preds <- predict(lasso_mod, newx = as.matrix(sapply(x_valid, as.numeric)), type = "response")
  pred_lasso <- prediction(preds,y_valid )
  roc_lasso <-performance(pred_lasso, "tpr", "fpr")
  auc_score <- performance(pred_lasso, measure = "auc")@y.values[[1]]
  auc_lasso[i] <- auc_score
}

plot(log10(grid), auc_lasso,
     ylab = "AUC Score", xlab = "Lambda",
     main = "Lasso: AUC Score vs Lambda")

# get best-performing lambda
best_validation_index <- which.max(auc_lasso)
best_lambda_lasso <- grid[best_validation_index]

# building model on best performing lambda
best_lasso <- glmnet(as.matrix(sapply(x_train, as.numeric)), as.matrix(y_train), family = "binomial", alpha = 1, lambda = best_lambda_lasso)
preds_lasso <- predict(best_lasso, newx = as.matrix(sapply(x_valid, as.numeric)), type = "response")
best_lasso_prediction <- prediction(preds_lasso, y_valid)
roc_best_lasso <-performance(best_lasso_prediction, "tpr", "fpr")
auc_lasso <- performance(best_lasso_prediction, measure = "auc")@y.values[[1]]
auc_lasso


#============================ 6. Random Forest ======================================


# Grid search cross validation for random forest

grid_search_rf <- function(){
  #three hyperparameters can possibly really change predictive performance of random forest (although maybe not)
  # you can add more hyperparameters here
  mtry_choose <- c(15,20,30,40,50,76)
  ntrees_choose <- c(100,200,300,500,600)
  
  
  # Initialize an empty dataframe to store results
  results <- data.frame(mtry = numeric(),
                        num.trees = numeric(),
                        auc = numeric(),
                        auc_train = numeric(),
                        stringsAsFactors = FALSE) 
  
  #nested loops to tune these three parameters
  print('mtry, num.trees, auc_score, auc_train')
  for(i in c(1:length(mtry_choose))){
    for(j in c(1:length(ntrees_choose))){
      thismtry <- mtry_choose[i]
      thisntrees <- ntrees_choose[j]
      
      
      inner_rf.mod <- ranger(x = x_train, y = y_train,
                             mtry=thismtry, num.trees=thisntrees,
                             importance="impurity",
                             probability = TRUE)
      
      inner_preds_rf <- predict(inner_rf.mod, data=x_valid)$predictions
      inner_rf_classifications <- ifelse(inner_preds_rf[,2]>0.5, 1, 0)
      inner_prediction_rf <- prediction(inner_rf_classifications, y_valid)
      inner_roc_rf <-performance(inner_prediction_rf, "tpr", "fpr")
      inner_auc_rf <- performance(inner_prediction_rf, measure = "auc")@y.values[[1]]
      
      #training
      inner_preds_rf_tr <- predict(inner_rf.mod, data=x_train)$predictions
      inner_rf_classifications_tr <- ifelse(inner_preds_rf_tr[,2]>0.5, 1, 0)
      inner_prediction_rf_tr <- prediction(inner_rf_classifications_tr, y_train)
      inner_roc_rf_tr <-performance(inner_prediction_rf_tr, "tpr", "fpr")
      inner_auc_rf_tr <- performance(inner_prediction_rf_tr, measure = "auc")@y.values[[1]]
      
      
      #print the performance for every combination
      print(paste(thismtry, thisntrees, inner_auc_rf,inner_auc_rf_tr, sep = ", "))
      # Append the results to the dataframe
      results <- rbind(results, data.frame(mtry = thismtry,
                                           num.trees = thisntrees,
                                           auc = inner_auc_rf,
                                           auc_train = inner_auc_rf_tr))
      
    }
  }
  return(results)
}

rf_results <- grid_search_rf()
rf_results[which.max(rf_results$auc),]

rf.mod <- ranger(x = x_train, y = y_train,
                 mtry=15, num.trees=500,
                 importance="impurity",
                 probability = TRUE)

preds_rf <- predict(rf.mod, data=x_valid)$predictions
rf_classifications <- ifelse(preds_rf[,2]>0.5, 1, 0)
prediction_rf <- prediction(rf_classifications, y_valid)
roc_rf <-performance(prediction_rf, "tpr", "fpr")
auc_rf <- performance(prediction_rf, measure = "auc")@y.values[[1]]
auc_rf

# Fitting Curve for Random Forest

plot_rf <- rf_results %>%
  filter(mtry == 76)

plot(plot_rf$num.trees,plot_rf$auc,type = 'o',col = "blue", 
     lwd= 2,xlab = "Number of Trees",
     ylab = "Validation AUC Score",
     main = "Random Forest: Validation Fitting Curve")

plot(plot_rf$num.trees,plot_rf$auc_train,type = 'o',col = "red",
     lwd= 2,xlab = "Number of Trees", ylab = "Training AUC Score",
     main = "Random Forest: Training Fitting Curve")



# Test data preprocessing

data_test <- data_prep(test_x)

# Text Mining on Amenities Test

it_test_amenities <- itoken(data_test$amenities, 
preprocessor = tolower, #preprocessing by converting to lowercase
tokenizer = cleaning_tokenizer, 
ids = data_test$id, 
progressbar = FALSE)

dtm_test_amenities <- create_dtm(it_test_amenities, vectorizer_amenities)

dtm_test_tfidf_amen <- fit_transform(dtm_test_amenities, tfidf_amenities)

# represented as a regular dataframe
amenities_test <- data.frame(as.matrix(dtm_test_tfidf_amen))


# Text Mining on host_verifications test

it_test_verif <- itoken(data_test$host_verifications, 
                        preprocessor = tolower, #preprocessing by converting to lowercase
                        tokenizer = cleaning_tokenizer, 
                        ids = data_test$id, 
                        progressbar = FALSE)


dtm_test_verif <- create_dtm(it_test_verif, vectorizer_verif)


dtm_test_tfidf_verif <- fit_transform(dtm_test_verif, tfidf_verif)

# represented as a regular dataframe
verification_test <- data.frame(as.matrix(dtm_test_tfidf_verif))


# Text Mining on house_rules test

it_test_rules <- itoken(data_test$house_rules, 
                        preprocessor = tolower, #preprocessing by converting to lowercase
                        tokenizer = cleaning_tokenizer2, 
                        ids = data_test$id, 
                        progressbar = FALSE)

dtm_test_rules <- create_dtm(it_test_rules, vectorizer_rules)

dtm_test_tfidf_rules <- fit_transform(dtm_test_rules, tfidf_rules)

# represented as a regular dataframe
rules_test <- data.frame(as.matrix(dtm_test_tfidf_rules))

rules_test <- rules_test %>%
  mutate(no_rules = NA.) %>%
  select(-NA.)

# Combining the text features with the main test data

am_verif_test <- cbind(amenities_test,verification_test)
am_trans_test <- cbind(am_verif_test, rules_test)
data_test2 <- cbind(data_test, am_trans_test)


# Selecting features
test_features <- data_test2 %>%
  select(accommodates, bedrooms,beds,cancellation_policy,
         cleaning_fee,host_total_listings_count,price,ppp_ind,
         price_per_person, property_category, bed_category, 
         bathrooms, extra_people, host_acceptance, 
         host_response,host_response_time,availability_30,
         availability_60,availability_90, availability_365,
         num_of_features, minimum_nights,market,host_since,
         first_review,host_is_superhost,instant_bookable, 
         latitude, longitude, guests_included,wireless.internet : no_rules)

# Creating dummy variables for test data
dummy_test <- dummyVars( ~ . , data=test_features, fullRank = TRUE)
test_dummy <- data.frame(predict(dummy_test, newdata = test_features))

# Removing some unwanted features
test_dummy <- test_dummy %>%
  select(-c(carbon.monoxide.detector,essentials,google,
            jumio,facebook,linkedin,keep,parti,
            home,pet,hous,pm,pleas,allow,leav))


# Training our best model - XGboost on the whole data
x_train_sub <- airbnb_dummy %>%
  select(-high_booking_rate.YES)

y_train_sub <- airbnb_dummy$high_booking_rate.YES 

dtrain_sub = xgb.DMatrix(as.matrix(sapply(x_train_sub, as.numeric)), label=as.matrix(y_train_sub))
bst_sub <- xgboost(data = dtrain_sub,
                   max.depth = 2,
                   eta = 0.5,
                   nrounds = 600,
                   scale_pos_weight = 3.9,
                   early_stopping_rounds = 5,
                   objective = "binary:logistic")

# Predictions on test data
bst_pred_sub <- predict(bst_sub, as.matrix(sapply(test_dummy, as.numeric)))

# checking if final prediction contains any null values
table(is.na(bst_pred_sub))

# creating outputs in the correct format
write.table(bst_pred_sub, "high_booking_rate_group21.csv", row.names = FALSE)

