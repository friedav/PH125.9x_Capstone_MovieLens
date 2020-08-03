#!/usr/bin/env Rscript
#
# R script implementing a solution to the MovieLens project
# by Friederike David

#### Prerequisits ####

# employed packages
pkgs <- c(
  "here",         # locate files within project
  "tidyverse",    # load multiple 'tidyverse' packages in a single step
  "stringr",      # string operations
  "data.table",   # fast reading and writing of tabular data
  "lubridate",    # date/time operations (masks here::here()!)
  "caret",        # classification and regression training
  "rpart"         # recursive partitioning and regression trees
)
lapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = T)) 
    install.packages(pkg, repos = "http://cran.us.r-project.org")
})

# plotting theme
theme_set(theme_minimal())

# R version (to use correct set.seed call)
rver <- paste0(sessionInfo()$R.version$major, sessionInfo()$R.version$minor) %>% 
  str_remove_all("\\.") %>% as.integer()


#### Create edx set, validation set ####
# (Code in this section copied and modified from course materials)
#
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

if (!file.exists(here::here("data/input_data.RData"))) {
  
  dl <- here::here("data/movielens_ml-10m.zip")
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  if (rver < 360) {set.seed(1)} else {set.seed(1, sample.kind = "Rounding")}
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  save(edx, validation, file = here::here("data/input_data.RData"))
  
} else {
  load(here::here("data/input_data.RData"))
}


#### Explore and prepare training set ####

# structure of data set
str(edx)

# number of movies and users
edx %>% summarize_at(c("userId", "movieId", "title"), ~ unique(.x) %>% length())

# available genres
edx %>% group_by(genres) %>% summarize(n = n()) %>% arrange(desc(n))
str_split(edx$genres, "\\|") %>% unlist() %>% table()

# add publication dates extracted from title and rating dates to data frame
edx <- edx %>%
  mutate(publication_year = str_extract(title, "(?<=\\()[0-9]{4}(?=\\)$)"),
         rating_date = as_datetime(timestamp),
         rating_year = year(rating_date),
         rating_weekday = wday(rating_date, label = T, week_start = 1))

# plot number of ratings by publication year
edx %>%
  group_by(movieId) %>%
  summarize(n = n(), year = unique(publication_year)) %>%
  ggplot(aes(year, n)) +
  geom_point() +
  scale_y_log10() +
  scale_x_discrete(breaks = edx$publication_year %>% unique() %>% str_subset("0$")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot distribution of rating means by publication year
# (e.g. do older movies have worse ratings?)
edx %>%
  group_by(movieId) %>%
  summarize(year = unique(publication_year),
            avg_rating = mean(rating)) %>%
  ggplot(aes(year, avg_rating)) +
  geom_boxplot() +
  scale_x_discrete(breaks = edx$publication_year %>% unique() %>% str_subset("0$")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot rating by genre
edx %>%
  group_by(genres) %>%
  filter(n() > 10000) %>%
  summarize(avg = mean(rating),
            sd = sd(rating),
            se = sd(rating)/sqrt(n())) %>%
  arrange(avg) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) +
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# plot rating by weekday
edx %>%
  group_by(rating_weekday) %>%
  summarize(mean = mean(rating)) %>%
  ggplot(aes(rating_weekday, mean)) +
  geom_point()


#### Transform data ####


#### Build prediction model ####

# # models to consider for ensemble prediction
# models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess",
#             "multinom", "qda", "rf", "adaboost")
#
# # fit individual models
# fits <- lapply(models, function(model) {
#   print(model)
#   train(rating ~ ., data = edx)
# }) %>% set_names(models)
#
#
#
# # TODO adapt for quantitative outcome
#
# # make predictions and calculate accuracy within training set to select well
# # performing models to include in ensemble prediction
# train_predictions <- sapply(fits, predict, edx)
# train_accuracies <- apply(predictions, 2, function(pred_rating) {
#   mean(edx$rating == pred_rating)
# })
#
# models_keep <- training_accuracies >= 0.8
# ensemble_pred <- apply(predictions[,models_keep], 1,
#                        function(i) names(sort(table(i), decreasing = T))[1])
# mean(mnist_27$test$y == ensemble_pred)





#### Validate model on test set ####

# predict ratings for validation set
predictions <- NA

# calculate accuracy of predictions
accuracy <- NA
  
  