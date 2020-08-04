#' ---
#' title: "HarvardX: PH125.9x Capstone - MovieLens Project"
#' author: "Friederike David"
#' date: "`r Sys.Date()`"
#' output: 
#'   pdf_document:
#'     toc: true
#'     toc_depth: 2
#' ---
#' 
#'   
#'     
#' *This project is part of the*
#' *[HarvardX's Data Science Capstone](https://www.edx.org/course/data-science-capstone) course,*
#' *which is the last out of nine courses within the*
#' *[HarvardX's Data Science Professional Certificate](https://www.edx.org/professional-certificate/harvardx-data-science).*
#' 
#' 
## ----setup, include=FALSE--------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = T, warning = F,  message = F, 
                      fig.align = "center", fig.width = 9)

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

#' 
#' 
#' # Introduction
#' 
#' Movie recommendation systems are a common application of machine learning models
#' that aim to predict how a given user would rate a given movie in order to
#' generate recommendations for movies a given user is likely to enjoy.
#' Such recommendation systems are employed for example by streaming services like
#' Netflix to enhance the user experience and therefore well-preforming models can
#' create considerable business value.  
#' 
#' In this project, a movie recommendation system based on the 
#' [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/) is 
#' implemented, which predicts movie recommendations for a given movie and user 
#' within the dataset from a set of input features.
#' The dataset consists of about 10 million movie ratings with additional 
#' information e.g. on movie genre and rating datetime, which are used to develop a
#' machine learning model as detailed in the methods section.  
#' 
#' Briefly, as first step the validation subset that is exclusively used for the
#' final model evaluation is split from the input dataset. The remaining data 
#' points are explored and cleaned before deciding on a regularized additive model
#' as modeling approach and then further split into a training and test set for 
#' model optimization.
#' Finally, model performance is evaluated in the results section based on the
#' validation set.
#' Both for model development and evaluation, the root mean squared error (RMSE) 
#' between predicted and true ratings is used as loss function.
#' 
#' 
#' # Methods
#' 
#' ## Data input
#' 
#' The publicly available 
#' [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/)
#' is downloaded and split into a validation set (`validation`, ~10%) and a set for
#' model development (`edx`, ~90%) using a modified version of the R code provided
#' in the course materials.
#' For computational efficiency when re-running the analysis, the downloaded 
#' dataset and prepared data objects are saved in `./data/` and re-used if already 
#' existing.  
#' 
## ----data_input, include=F-------------------------------------------------------------------------------------------------------------------------------------
if (!file.exists(here::here("data/input_data.RData"))) {
  
  # stable download url for MovieLens 10M dataset
  url <- "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
  
  dl <- here::here("data/movielens_ml-10m.zip")
  if (!file.exists(dl)) download.file(url, dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% 
    mutate(movieId = as.numeric(levels(movieId))[movieId],
           title = as.character(title),
           genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # validation set will be 10% of MovieLens data
  if (rver < 360) {set.seed(1)} else {set.seed(1, sample.kind = "Rounding")}
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = .1, list = F)
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

#' 
#' 
#' ## Data exploration and feature engineering
#' 
#' For developing a well-performing machine learning model, it is essential to
#' explore the dataset before deciding on a modeling approach.
#' Therefore, in this section the structure and properties of the `edx` set are
#' examined and potentially relevant input features are prepared.
#' 
#' 
#' ### Overall structure
#' 
## ----data_exploration_overall, echo=F--------------------------------------------------------------------------------------------------------------------------
# structure of data set
glimpse(edx)

# number of unique elements, e.g. movies and users
n <- edx %>% summarize_all(~ unique(.x) %>% length())
genres.single <- str_split(edx$genres, "\\|") %>% unlist() %>% unique()

# # year of publication in parentheses included at the end of every title?
# all(str_detect(edx$title, "(?<=\\()[0-9]{4}(?=\\)$)"))

#' 
#' The training dataset contains `r ncol(edx)` columns
#' (`r paste0(colnames(edx), collapse = ", ")`) and `r nrow(edx)` rows.
#' It comprises `r n$userId` user IDs and `r n$movieId` movie IDs/`r n$title`
#' titles as well as `r length(genres.single)` genre categories and `r n$genres`
#' combinations of one or more genres.
#' Each title also includes the publication year in parentheses.  
#' 
#' Since the datetime of ratings is only given as integer timestamp and the
#' publication year is part of the title, inferred variables are added as potential
#' additional features. 
#' 
## ----features_dates--------------------------------------------------------------------------------------------------------------------------------------------
# add publication date extracted from title and rating date, year and weekday
edx <- edx %>%
  mutate(publication_year = str_extract(title, "(?<=\\()[0-9]{4}(?=\\)$)"),
         rating_date = as_datetime(timestamp),
         rating_year = year(rating_date),
         rating_weekday = wday(rating_date, label = T, week_start = 1))

#' 
#' 
#' 
#' ### Ratings
#' 
## ----data_exploration_ratings, echo=F, fig.height=3------------------------------------------------------------------------------------------------------------
# range and distribution of ratings
ggplot(edx, aes(x = rating)) +
  geom_bar() +
  labs(x = "Rating", y = "Count")

#' 
#' Movie ratings are recorded within a `r max(edx$rating)`-star system that allows 
#' for half-star and full-star ratings. 0-star ratings are not allowed.
#' In general, full-star ratings are more common than half-star ratings, although
#' the overall distribution is similar with 4 stars as most frequent full-star
#' rating and 3.5 stars as most frequent half-star rating.
#' 
#' 
#' ### Movies and users
#' 
#' Trends regarding movie-level and user-level data distribution are visualized in
#' the following plots. 
#' To evaluate sparsity within the user ID x movie ID rating matrix, a heatmap of 
#' ratings for a random subsample of 250 users and 250 movies is shown.  
#' 
## ----data_exploration_sparsity, fig.height=7, echo=F-----------------------------------------------------------------------------------------------------------
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

# user statistics: fraction of rated movies per user
uprop <- group_by(edx, userId) %>% summarize(prop = n()/n$movieId) %>% pull(prop)

# movie statistics: fraction of rating users per movie
mprop <- group_by(edx, movieId) %>% summarize(prop = n()/n$userId) %>% pull(prop)

#' 
#' 
#' A wide range in the number of ratings per movie and per user is present with 
#' some users having rated up to `r max(uprop * 100) %>% round(digits = 2)` % of 
#' all available movies and some movies having been rated by up to
#' `r max(mprop * 100) %>% round(digits = 2)` % of all available users.  
#' 
#' 
## ----data_exploration_ratings_movie, echo=F, fig.height=3------------------------------------------------------------------------------------------------------
# average rating by number of ratings per movie
p1 <- edx %>%
  group_by(movieId) %>%
  summarize(n = n(), avg_rating = mean(rating)) %>%
  ggplot(aes(n, avg_rating)) +
  geom_point() +
  geom_density_2d() +
  ylim(c(0, 5)) +
  scale_x_log10() +
  labs(x = "Number of ratings per movie", y = "Average rating by movie")
ggMarginal(p1, type = "boxplot")

#' 
#' 
## ----data_exploration_ratings_user, echo=F, fig.height=3-------------------------------------------------------------------------------------------------------
# average rating by number of ratings per user
p2 <- edx %>%
  group_by(userId) %>%
  summarize(n = n(), avg_rating = mean(rating)) %>%
  ggplot(aes(n, avg_rating)) +
  geom_point() +
  geom_density_2d() +
  ylim(c(0, 5)) +
  scale_x_log10() +
  labs(x = "Number of ratings per user", y = "Average rating by user")
ggMarginal(p2, type = "boxplot")

#' 
#' 
#' On average, movies with a higher number of ratings, i.e. popular movies, tend to
#' have a higher rating than movies with a low number of ratings, for which a broad
#' range of average ratings is observed.
#' In contrast, on a per-user level the relation between the average rating and the
#' number of ratings per user is not as strong, even though users with a very high
#' number of rated movies tend to have a below-average mean rating. 
#' Unsurprisingly, the variability of rating averages per movie or per user
#' decreases with the number of ratings.    
#' 
#' 
#' ### Genres
#' 
#' The `genres` column in the dataset contains a set of one or more main genres per
#' movie. In the following plots, the properties of both individual main genres as
#' well as genre combinations are examined.  
#' 
## ----data_exploration_genres_main, echo=F, fig.height=4--------------------------------------------------------------------------------------------------------
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

#' 
#' 
## ----data_exploration_genres_comb, echo=F, fig.height=4--------------------------------------------------------------------------------------------------------
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

# table
genres.comb.df %>%
  arrange(desc(n)) %>%
  select("Genre combination" = genres, 
         "Number of ratings" = n,
         "Mean rating" = avg_rating) %>%
  head(n = 5) %>%
  knitr::kable(caption = "Top 5 most frequent genre combinations")

#' 
#' For the `r length(genres.single)` main genres, only a very small impact on
#' average movie ratings is visible, with rating means per genre ranging from
#' `r min(genres.df$avg_rating) %>% round(digits = 2)` for 
#' `r genres.df$main_genre[which.min(genres.df$avg_rating)]` to
#' `r max(genres.df$avg_rating) %>% round(digits = 2)` for 
#' `r genres.df$main_genre[which.max(genres.df$avg_rating)]`.
#' In contrast, a moderate effect of genre combinations on ratings is visible, with
#' group means ranging from 
#' `r min(genres.comb.df$avg_rating) %>% round(digits = 2)` for 
#' `r genres.comb.df$genres[which.min(genres.comb.df$avg_rating)]` to
#' `r max(genres.comb.df$avg_rating) %>% round(digits = 2)` for 
#' `r genres.comb.df$genres[which.max(genres.comb.df$avg_rating)]`.
#' Again, with lower numbers of ratings per genre combination, the variability in
#' average ratings increases.  
#' 
#' 
#' ### Publication year and rating datetime
#' 
#' Time-related trends are visualized in the following plots.
#' 
## ----data_exploration_time_rating, fig.height=3, echo=F--------------------------------------------------------------------------------------------------------
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

#' 
#' The rating year seems to have only a small impact on average movie ratings, 
#' especially when considering rating years with very few data points as rather
#' noisy outliers. 
#' The effect by rating weekday seems to be even smaller, with the mean of average
#' ratings per movie ranging from
#' `r group_by(rday.df, rating_weekday) %>% summarize(avg = mean(avg_rating)) %>% pull(avg) %>% min() %>% round(digits = 2)` 
#' to
#' `r group_by(rday.df, rating_weekday) %>% summarize(avg = mean(avg_rating)) %>% pull(avg) %>% max() %>% round(digits = 2)`
#' across all weekdays.  
#' 
#' 
## ----data_exploration_time_publication, fig.height=4, echo=F---------------------------------------------------------------------------------------------------
# relation between publication year and mean rating per movie
pubyear.df <- edx %>%
  group_by(movieId) %>%
  summarize(year = unique(publication_year),
            avg_rating = mean(rating))
ggplot(pubyear.df, aes(year, avg_rating)) +
  geom_boxplot() +
  scale_x_discrete(breaks = edx$publication_year %>% unique() %>% str_subset("0$")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Publication year", y = "Average rating by movie")

#' 
#' The effect of publication year on average movie ratings ranges from
#' `r group_by(pubyear.df, year) %>% summarize(avg = mean(avg_rating)) %>% pull(avg) %>% min() %>% round(digits = 2)` to
#' `r group_by(pubyear.df, year) %>% summarize(avg = mean(avg_rating)) %>% pull(avg) %>% max() %>% round(digits = 2)`
#' as rating means per publication year.
#' 
#' 
#' ## Model development
#' 
#' For both model development and evaluation, the root mean squared error (RMSE) 
#' with predicted ratings $\hat{Y}$, true ratings $Y$ and number of ratings $N$ 
#' is used as loss function:  
#' 
#' $$RMSE = \sqrt{\frac{1}{N}\sum \hat{Y} - Y}$$
#' 
## ----fun_rmse--------------------------------------------------------------------------------------------------------------------------------------------------
# function to calculate root mean squared error
RMSE <- function(Y_hat, Y) sqrt(mean((Y_hat - Y)^2))

#' 
#' 
#' From the preceding data exploration it can be assumed that both movie-specific
#' and user-specific effects are important for modeling movie ratings.
#' Further, the specified combination of movie genres seems to provide additional 
#' information on movie ratings and are thus included in the model as well.
#' Rating year, rating weekday and publication year, however, seem to have a rather
#' small impact on ratings and are therefore not included in the model.  
#' 
#' The resulting model for movie ratings $Y$ can be written as follows with the
#' overall average rating $\mu$ as baseline and level-specific feature effects $b$ 
#' for movie $m$, user $u$ and movie genre $g$ as well as an error term $\epsilon$:  
#' 
#' $$Y = \mu + b_m + b_u + b_g + \epsilon$$
#' 
#' 
#' As baseline, $\mu$ is estimated as the overall average rating.  
#' 
## ----mu_hat----------------------------------------------------------------------------------------------------------------------------------------------------
# estimate mean rating
mu_hat <- mean(edx$rating)

#' 
#' In a stepwise approach which reflects the assumed relevance of predictor 
#' variables, the individual effects $b$ for available factor levels are estimated
#' according to the following sequence of equations:
#' 
#' $$\hat{b}_m = \frac{1}{n_m}\sum{Y_m - \hat{\mu}}$$
#' $$\hat{b}_u = \frac{1}{n_u}\sum{Y_u - \hat{\mu} - \hat{b}_m}$$
#' $$\hat{b}_g = \frac{1}{n_g}\sum{Y_g - \hat{\mu} - \hat{b}_m - \hat{b}_u}$$
#' 
#' 
## ----coeff_basic-----------------------------------------------------------------------------------------------------------------------------------------------
# estimate model coefficients for movie, user and genre effects
movie_basic <- edx %>% 
  group_by(movieId) %>% summarize(b_m = mean(rating - mu_hat))

user_basic <- edx %>% left_join(movie_basic) %>% 
  group_by(userId) %>% summarize(b_u = mean(rating - mu_hat - b_m))

genre_basic <- edx %>% left_join(movie_basic) %>% left_join(user_basic) %>% 
  group_by(genres) %>% summarize(b_g = mean(rating - mu_hat - b_m - b_u))

#' 
#' 
#' Since for some movies, users and genres only very few ratings are 
#' available, regularization of coefficients is additionally used to improve model
#' performance.
#' Here, the following function will be minimized to obtain the best value for 
#' the penalty parameter $\lambda$:
#' 
#' $$\frac{1}{N}\sum{(Y - \mu - b_m - b_u - b_g)^2} + \lambda (\sum{b_m^2} + \sum{b_u^2} + \sum{b_g^2})$$
#' 
#' As before in the basic estimation of coefficients, the stepwise approach
#' is employed.
#' Since the penalty $\lambda$ is a tunable parameter, the `edx` set is split into
#' a training set `train` and a testing set `test` to chose the optimal $\lambda$ 
#' that minimizes the RMSE in the testing set.
#' Using the `train` set to calculate regularized coefficients at different values
#' for $\lambda$ and the `test` set to calculate the corresponding RMSE, the 
#' optimal value for $\lambda$ that minimizes the RMSE on the test set is then
#' selected.
#' 
#' 
## ----split_edx-------------------------------------------------------------------------------------------------------------------------------------------------
# split edx set into training and testing set for model optimization
if (rver < 360) {set.seed(1)} else {set.seed(1, sample.kind = "Rounding")}
train_index <- createDataPartition(edx$rating, p = 0.9, list = F)

train <- edx[train_index,]
tmp <- edx[-train_index,]

# make sure userId and movieId in test set are also in train set
test <- semi_join(tmp, train, by = "movieId") %>% semi_join(train, by = "userId")

# add rows removed from test set back into train set
train <- rbind(train, anti_join(tmp, test))

#' 
#' 
## ----lambda_tuning---------------------------------------------------------------------------------------------------------------------------------------------
# calculate RSME at different lambdas (regularizing movie, user and genre effects)
lambdas <- seq(0, 10, 0.5)
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

#' 
#' 
## ----lambda, echo=F, fig.height=3------------------------------------------------------------------------------------------------------------------------------
data.frame(Lambda = lambdas,
           RMSE = rmse_reg) %>% 
  ggplot(aes(Lambda, RMSE)) +
  geom_point()

# get coefficients for lambda that minimizes the RSME
lambda <- lambdas[which.min(unlist(rmse_reg))]

#' 
#' As the plot demonstrates, the minimum RMSE is obtained at $\lambda =$ `r lambda`.
#' With this optimal value, the final regularized model coefficients are calculated.  
#' 
## ----coeff_regularized-----------------------------------------------------------------------------------------------------------------------------------------
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

#' 
#' 
#' # Results
#' 
#' After fitting the basic and regularized models as described before in the
#' methods section, performance is evaluated by calculating the RMSE on the 
#' validation set.
#' For comparison purposes, the RMSE obtained when using the overall average rating
#' as naive prediction is also included.  
#' 
## ----modeling_evaluation---------------------------------------------------------------------------------------------------------------------------------------
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

#' 
#' 
## ----modeling_evaluation_kable, echo=F-------------------------------------------------------------------------------------------------------------------------
data.frame(Model = c("naive mean", 
                     "simple additive model", 
                     "regularized additive model"),
           RMSE = c(rmse_naive_mean, 
                    rmse_basic, 
                    rmse_regularized)) %>% 
  knitr::kable(caption = "RSME estimates for different modeling approaches")

#' 
#' 
#' Compared to the naive model, in which all ratings are predicted as the overall
#' average, with an RMSE of `r  round(rmse_naive_mean, digits = 5)`, the simple 
#' additive model, in which movie effects, user effects and genre effects are 
#' considered, yields a considerably lower RMSE of `r round(rmse_basic, digits = 5)`.
#' The regularization of coefficients leads to a further improvement of model
#' performance, however, with a RMSE of `r round(rmse_regularized, digits = 5)` 
#' this improvement is rather minor.  
#' 
#' 
#' # Conclusion
#' 
#' In this project, a movie recommendation system was implemented using a 
#' regularized additive linear regression model with movie effects, user effects 
#' and genre effects as predictors. 
#' A final RMSE of `r round(rmse_regularized, digits = 5)` on the validation set 
#' was obtained, which satisfies the requirements specified in the course 
#' instructions.  
#' 
#' Within the employed model architecture, the inclusion of additional features 
#' like publication year or rating year may have yielded an even lower RMSE, 
#' however, since the improvement was expected to be small, a more simplistic model
#' was favoured in this context.  
#' 
#' Considering that this project is part of the 
#' [HarvardX's Data Science Professional Certificate](https://www.edx.org/professional-certificate/harvardx-data-science),
#' the selection of a modeling approach was to some extend based on the lessons
#' learned in the preceding
#' [HarvardX's Data Science Machine Learning](https://www.edx.org/course/data-science-machine-learning) 
#' course.
#' Thus, further improvements in model performance may be achieved by employing
#' more advanced approaches not extensively covered in the course such as matrix 
#' factorization. This approach makes use of patterns within the dataset but is
#' also more expensive in model development.   
#' 
#' 
#' # Session info
#' 
## ----session_info, echo=F--------------------------------------------------------------------------------------------------------------------------------------
sessionInfo()

#' 
