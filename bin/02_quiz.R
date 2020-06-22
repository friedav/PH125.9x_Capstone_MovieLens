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

