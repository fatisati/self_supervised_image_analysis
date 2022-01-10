airquality = read.csv('airquality.csv',header=TRUE, sep=",")

explain_data <- function(data){
  print(str(data))
  print(head(data))
  print(tail(data))
  print(summary(data))
}

# margin of the grid(mar), no of rows and columns(mfrow)
# whether a border is to be included(bty)
# and position of the labels(las: 1 for horizontal, las: 0 for vertical)
# example(plot) runs the demo of the plot directly in the console.
grid_chart <- function(las=1){
  par(mfrow=c(3,3), mar=c(2,5,2,1), las=las, bty="n")
  plot(airquality$Ozone)
  plot(airquality$Ozone, airquality$Wind)
  plot(airquality$Ozone, type= "c")
  plot(airquality$Ozone, type= "s")
  plot(airquality$Ozone, type= "h")
  barplot(airquality$Ozone, main = 'Ozone Concenteration in air',xlab = 'ozone levels', col='green',horiz = TRUE)
  hist(airquality$Solar.R)
  boxplot(airquality$Solar.R)
  boxplot(airquality[,0:4], main='Multiple Box plots')
}

plot(airquality$Ozone)
plot(airquality$Ozone, airquality$Wind)
plot(airquality)

#types: p: points, l: lines,b: both, h: high density vertical lines
#read more ?plot()
plot(airquality$Ozone, type= "h")

#labels and titles
plot(airquality$Ozone, xlab = 'ozone Concentration', ylab = 'No of Instances', main = 'Ozone levels in NY city', col = 'green')

# Horizontal bar plot
barplot(airquality$Ozone, main = 'Ozone Concenteration in air',xlab = 'ozone levels', col= 'green',horiz = TRUE)

# Vertical bar plot
barplot(airquality$Ozone, main = 'Ozone Concenteration in air',xlab = 'ozone levels', col='red',horiz = FALSE)

hist(airquality$Solar.R)

hist(airquality$Solar.R, main = 'Solar Radiation values in air',xlab = 'Solar rad.', col='red')

#Single box plot
boxplot(airquality$Solar.R)

# Multiple box plots
boxplot(airquality[,0:3], main='Multiple Box plots')
grid_chart(1)
