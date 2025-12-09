install.packages("tidyverse")
install.packages("scales")

library(tidyverse)
library(scales)

setwd("~/Desktop/GMU/CS504")

person <- read_csv("Motor_Vehicle_Collisions_-_Person.csv")

person_clean <- person %>%
  rename(
    age   = PERSON_AGE,
    sex   = PERSON_SEX,
    injury = PERSON_INJURY
  ) %>%
  # Filter out missing or obviously bad ages
  filter(!is.na(age),
         age >= 0,
         age <= 99) %>%
  # Keep main sex categories (you can keep "U" or "Unknown" if present)
  mutate(
    sex = case_when(
      sex %in% c("M", "MALE") ~ "Male",
      sex %in% c("F", "FEMALE") ~ "Female",
      TRUE ~ "Unknown"
    ),
    # Define injury vs non-injury (adjust values if needed)
    injured_flag = if_else(injury %in% c("Injured", "Killed", "Fatal Injury"),
                           1L, 0L)
  )

person_clean <- person_clean %>%
  mutate(
    age_group = cut(
      age,
      breaks = c(-Inf, 17, 24, 44, 64, Inf),
      labels = c("0–17", "18–24", "25–44", "45–64", "65+"),
      right = TRUE
    )
  )

summary_age_gender <- person_clean %>%
  filter(!is.na(age_group)) %>%
  group_by(age_group, sex) %>%
  summarise(
    n_people   = n(),
    n_injured  = sum(injured_flag, na.rm = TRUE),
    injury_rate = n_injured / n_people,
    .groups = "drop"
  )

print(summary_age_gender)

ggplot(summary_age_gender,
       aes(x = age_group, y = injury_rate, fill = sex)) +
  geom_col(position = position_dodge()) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    name = "Injury Rate (%)"
  ) +
  scale_fill_brewer(palette = "Set1", name = "Gender") +
  labs(
    title = "Injury Rate by Age Group and Gender in NYC Collisions",
    x = "Age Group"
  ) +
  theme_minimal(base_size = 12)

ggsave("age_gender_injury_rate_nyc.png",
       width = 8, height = 5, dpi = 300)
