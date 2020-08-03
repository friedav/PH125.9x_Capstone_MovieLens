#!/usr/bin/env Rscript
#
# Quiz: MovieLens Dataset
# solution by Friederike David

#### Prerequisits ####

# employed packages
pkgs <- c(
  "here",         # locate files within project
  "tidyverse",    # load multiple 'tidyverse' packages in a single step
  "data.table",   # fast reading and writing of tabular data
  "caret",        # classification and regression training
)
lapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = T)) 
    install.packages(pkg, repos = "http://cran.us.r-project.org")
})

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
  save(edx, validation, file = here("data/input_data.RData"))
  
} else {
  load(here("data/input_data.RData"))
}


##### Quiz: MovieLens Dataset ####

# Q1 How many rows and columns are there in the edx dataset?
str(edx)

# Q2 How many zeros were given as ratings in the edx dataset?
# How many threes were given as ratings in the edx dataset?
sum(edx$rating == 0)
sum(edx$rating == 3)

# Q3 How many different movies are in the edx dataset?
edx$movieId %>% unique() %>% length() # right answer: 10677
edx$title %>% unique() %>% length() # however: 10676 titles

# Q4 How many different users are in the edx dataset?
edx$userId %>% unique() %>% length()

# Q5 How many movie ratings are in each of the following genres in the edx dataset?
sum(str_detect(edx$genres, "Drama"))
sum(str_detect(edx$genres, "Comedy"))
sum(str_detect(edx$genres, "Thriller"))
sum(str_detect(edx$genres, "Romance"))

# Q6 Which movie has the greatest number of ratings?
edx %>% 
  group_by(title) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))

# Q7 What are the five most given ratings in order from most to least?
edx %>% 
  group_by(rating) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))

# Q8 True or False: In general, half star ratings are less common than whole star 
# ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
edx %>% 
  summarize(full_star = sum(rating %in% 0:5),
            half_star = sum(rating %in% seq(0.5, 5.5)),
            answer = half_star < full_star) 

