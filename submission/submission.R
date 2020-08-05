#!/usr/bin/env Rscript
#
# title: "HarvardX: PH125.9x Capstone - MovieLens Project"
# author: "Friederike David"
#
# This script is a collection of code snippets from the accompanying Rmd file
# implementing a movie recommendation system based on the 10M MovieLens dataset.
# The code may include additional/modified comments but is in principle 
# equivalent to snippets in the Rmd file.

#### Setup ####

# employed packages
pkgs <- c(
  "here",         # locate files within project
  "tidyverse",    # load multiple 'tidyverse' packages in a single step
  "stringr",      # string operations
  "data.table",   # fast reading and writing of tabular data
  "lubridate",    # date/time operations (masks here::here()!)
  "caret",        # classification and regression training
  "glmnet",       # compute penalized regression
  "rpart",        # recursive partitioning and regression trees
  "patchwork",    # plotting
  "scales",       # plotting
  "ggrepel",      # repel overlapping text labels
  "ggExtra"       # marginal boxplots
)
lapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = T)) 
    install.packages(pkg, repos = "http://cran.us.r-project.org")
})

# project setup: use "<projectdir>/data" to store reusable input data
dir.create(here::here("data"), showWarnings = F)

# plotting theme
theme_set(theme_minimal())

# R version (to use correct set.seed call)
rver <- paste0(sessionInfo()$R.version$major, sessionInfo()$R.version$minor) %>% 
  str_remove_all("\\.") %>% as.integer()


#### Data input ####
# for computational efficiency when re-running the analysis, the downloaded 
# dataset and prepared data objects are saved in `./data/` and re-used if 
# already existing

if (!file.exists(here::here("data/input_data.RData"))) {
  
  # stable download url for MovieLens 10M dataset
  url <- "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
  
  dl <- here::here("data/movielens_ml-10m.zip")
  if (!file.exists(dl)) download.file(url, dl)
  
  ratings <- fread(text = gsub("::", "\t", 
                               readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), 
                            "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% 
    mutate(movieId = as.numeric(levels(movieId))[movieId],
           title = as.character(title),
           genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # validation set will be 10% of MovieLens data
  if (rver < 360) {set.seed(1)} else {set.seed(1, sample.kind = "Rounding")}
  test_index <- createDataPartition(y = movielens$rating, p = .1, list = F)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(url, dl, ratings, movies, test_index, temp, movielens, removed)
  save(edx, validation, file = here::here("data/input_data.RData"))
  
} else {
  load(here::here("data/input_data.RData"))
}


#### Data exploration and feature engineering ####

## Overall structure

# structure of data set
glimpse(edx)

# number of unique elements, e.g. movies and users
n <- edx %>% summarize_all(~ unique(.x) %>% length())
genres.single <- str_split(edx$genres, "\\|") %>% unlist() %>% unique()

# # year of publication in parentheses included at the end of every title?
# all(str_detect(edx$title, "(?<=\\()[0-9]{4}(?=\\)$)"))


## Time-related features

# add publication date extracted from title and rating date, year and weekday
edx <- edx %>%
  mutate(publication_year = str_extract(title, "(?<=\\()[0-9]{4}(?=\\)$)"),
         rating_date = as_datetime(timestamp),
         rating_year = year(rating_date),
         rating_weekday = wday(rating_date, label = T, week_start = 1))


## Ratings

# range and distribution of ratings
ggplot(edx, aes(x = rating)) +
  geom_bar() +
  labs(x = "Rating", y = "Count")


## Movies and users

# visualize rating matrix on a small random subset of users and movies
edx %>%
  filter(movieId %in% sample(movieId, 250)) %>%
  filter(userId %in% sample(userId, 250)) %>%
  mutate_at(vars(movieId, userId), ~ as.character(.x)) %>%
  ggplot(aes(movieId, userId, fill = rating)) +
  geom_tile() +
  labs(x = "Movies", y = "Users", fill = "Rating") +
  theme(axis.text = element_blank(),
        legend.position = "bottom")

# user statistics: fraction/number of rated movies and average rating per user
ustat <- group_by(edx, userId) %>% 
  summarize(prop = n()/n$userId, 
            n = n(), 
            avg_rating = mean(rating))

# movie statistics: fraction/number of rating users and average rating per movie
mstat <- group_by(edx, movieId) %>% 
  summarize(prop = n()/n$userId, 
            n = n(), 
            avg_rating = mean(rating))

# average rating by number of ratings per movie
p1 <- mstat %>%
  ggplot(aes(n, avg_rating)) +
  geom_point() +
  geom_density_2d() +
  ylim(c(0, 5)) +
  scale_x_log10() +
  labs(x = "Number of ratings per movie", y = "Average rating by movie")
ggMarginal(p1, type = "boxplot")

# average rating by number of ratings per user
p2 <- ustat %>%
  ggplot(aes(n, avg_rating)) +
  geom_point() +
  geom_density_2d() +
  ylim(c(0, 5)) +
  scale_x_log10() +
  labs(x = "Number of ratings per user", y = "Average rating by user")
ggMarginal(p2, type = "boxplot")


## Genres

# table of ratings by main genres
genres.df <- lapply(genres.single, function(genre) {
  edx %>%
    filter(str_detect(genres, genre)) %>%
    mutate(main_genre = genre) %>%
    select(main_genre, rating)
}) %>% bind_rows() %>%
  mutate(main_genre = reorder(main_genre, -rating, FUN = mean)) %>%
  group_by(main_genre) %>%
  summarize(n = n(), avg_rating = mean(rating))

pmain <- ggplot(genres.df, aes(n, avg_rating, label = main_genre)) +
  geom_point() +
  geom_text_repel() +
  ylim(c(0, 5)) +
  labs(x = "Number of ratings per main genre", y = "Average rating")
ggMarginal(pmain, type = "boxplot")

# available genre combinations
genres.comb.df <- edx %>% group_by(genres) %>%
  summarize(n = n(), avg_rating = mean(rating)) 

# average rating by number of ratings per genre combination
pcomb <- ggplot(genres.comb.df, aes(n, avg_rating)) +
  geom_point() +
  ylim(c(0, 5)) +
  scale_x_log10() +
  labs(x = "Number of ratings per genre combination", y = "Average rating")
ggMarginal(pcomb, type = "boxplot")

# table of most frequent genre combinations
genres.comb.df %>%
  arrange(desc(n)) %>%
  select("Genre combination" = genres, 
         "Number of ratings" = n,
         "Mean rating" = avg_rating) %>%
  head(n = 5) 


## Publication year and rating datetime

# average rating over time
ryear.df <- edx %>%
  mutate(rating_year = as.character(rating_year)) %>%
  group_by(movieId, rating_year) %>%
  summarize(avg_rating = mean(rating))
p1 <- ggplot(ryear.df, aes(rating_year, avg_rating)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(c(0, 5)) +
  labs(x = "Rating year", y = "Average rating by movie")

# average rating by weekday
rday.df <- edx %>%
  group_by(movieId, rating_weekday) %>%
  summarize(avg_rating = mean(rating))
p2 <- ggplot(rday.df, aes(rating_weekday, avg_rating)) +
  geom_boxplot() +
  ylim(c(0, 5)) +
  labs(x = "Rating weekday", y = "Average rating by movie")

(p1 + p2)

# relation between publication year and mean rating per movie
pubyear.df <- edx %>%
  group_by(movieId) %>%
  summarize(year = unique(publication_year),
            avg_rating = mean(rating))
ggplot(pubyear.df, aes(year, avg_rating)) +
  geom_boxplot() +
  scale_x_discrete(breaks = unique(edx$publication_year) %>% str_subset("0$")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Publication year", y = "Average rating by movie")


#### Model development ####
# fit a simple additive model and a regularized additive model both considering
# movie, user and genre effects as predictors

# function to calculate root mean squared error
RMSE <- function(Y_hat, Y) sqrt(mean((Y_hat - Y)^2))

# estimate mean rating as baseline
mu_hat <- mean(edx$rating)


## simple additive model

# estimate model coefficients for movie, user and genre effects
movie_basic <- edx %>% 
  group_by(movieId) %>% summarize(b_m = mean(rating - mu_hat))

user_basic <- edx %>% left_join(movie_basic) %>% 
  group_by(userId) %>% summarize(b_u = mean(rating - mu_hat - b_m))

genre_basic <- edx %>% left_join(movie_basic) %>% left_join(user_basic) %>% 
  group_by(genres) %>% summarize(b_g = mean(rating - mu_hat - b_m - b_u))


## regularized additive model

# split edx set into training and testing set for model optimization
if (rver < 360) {set.seed(1)} else {set.seed(1, sample.kind = "Rounding")}
train_index <- createDataPartition(edx$rating, p = 0.9, list = F)
train <- edx[train_index,]
tmp <- edx[-train_index,]

# make sure userId and movieId in test set are also in train set
test <- semi_join(tmp, train, by = "movieId") %>% semi_join(train, by = "userId")

# add rows removed from test set back into train set
train <- rbind(train, anti_join(tmp, test))

# calculate RSME at different lambdas regularizing movie, user and genre effects
lambdas <- seq(0, 10, 0.25)
rmse_reg <- sapply(lambdas, function(lambda) {
  
  # regularize movie effects
  movie_reg <- train %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu_hat)/(lambda + n()))
  
  # regularize user effects
  user_reg <- train %>% 
    left_join(movie_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - mu_hat - b_m)/(lambda + n()))
  
  # regularize genre effects
  genre_reg <- train %>% 
    left_join(movie_reg, by = "movieId") %>% 
    left_join(user_reg, by = "userId") %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - mu_hat - b_m - b_u)/(lambda + n()))
  
  # estimate RSME for test set
  test %>% 
    left_join(movie_reg, by = "movieId") %>% 
    left_join(user_reg, by = "userId") %>% 
    left_join(genre_reg, by = "genres") %>% 
    mutate(rating_hat = mu_hat + b_m + b_u + b_g) %>% 
    summarize(rmse = RMSE(rating, rating_hat)) %>% 
    pull(rmse)
})

# plot RMSE at different lambdas to visualize the position of optimum
data.frame(Lambda = lambdas,
           RMSE = rmse_reg) %>% 
  ggplot(aes(Lambda, RMSE)) +
  geom_point()

# get coefficients for lambda that minimizes the RSME
lambda <- lambdas[which.min(unlist(rmse_reg))]

# calculate regularized coefficients with selected lambda for the entire edx set
movie_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - mu_hat)/(lambda + n()))

user_reg <- edx %>% 
  left_join(movie_reg, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - mu_hat - b_m)/(lambda + n()))

genre_reg <- edx %>% 
  left_join(movie_reg, by = "movieId") %>% 
  left_join(user_reg, by = "userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating - mu_hat - b_m - b_u)/(lambda + n()))


#### Results ####

# estimate RMSE for naive model (overall mean)
rmse_naive_mean <- RMSE(validation$rating, mu_hat)

# estimate RMSE for basic model
rmse_basic <- validation %>% 
  left_join(movie_basic) %>% 
  left_join(user_basic) %>% 
  left_join(genre_basic) %>% 
  mutate(rating_hat = mu_hat + b_m + b_u + b_g) %>% 
  summarize(rmse = RMSE(rating, rating_hat)) %>% 
  pull(rmse)

# estimate RMSE for regularized model
rmse_regularized <- validation %>% 
  left_join(movie_reg, by = "movieId") %>% 
  left_join(user_reg, by = "userId") %>% 
  mutate(rating_hat = mu_hat + b_m + b_u) %>% 
  summarize(rmse = RMSE(rating, rating_hat)) %>% 
  pull(rmse)

# collect different RMSEs in common data frame for easier comparison
data.frame(Model = c("naive mean", 
                     "simple additive model", 
                     "regularized additive model"),
           RMSE = c(rmse_naive_mean, 
                    rmse_basic, 
                    rmse_regularized))


#### Session info ####
sessionInfo()
