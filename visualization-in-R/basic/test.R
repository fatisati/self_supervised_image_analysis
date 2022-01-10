airquality = read.csv('airquality.csv',header=TRUE, sep=",")
arr = c(1,2,3,4)
hist(arr)

a = c(1,2,3)
b = c(2,3,4)

df = data.frame(a,b)
df$a
class(df$a)

bool_arr0 = c(TRUE, FALSE)
bool_arr1 = c(TRUE, "false")
bool_arr1 = as.logical(toupper(bool_arr1))

library(ggplot2)

# Create a function to print squares of numbers in sequence.
my_print <- function(a) {
 print(a)
}	
