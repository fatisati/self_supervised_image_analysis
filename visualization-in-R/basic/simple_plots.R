# Libraries
library(ggplot2)
library(dplyr)

# Load dataset from github
data <- read.csv('../../../results/tables/sample-log.csv')


ggplot(data) +
geom_line(aes(x=epoch, y=val_recall)) +
geom_line(aes(x=epoch, y=val_precision))