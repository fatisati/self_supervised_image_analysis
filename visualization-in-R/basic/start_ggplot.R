library(ggplot2)
ggplot(data)  # if only the dataset is known.
ggplot(diamonds, aes(x=carat))  # if only X-axis is known. The Y-axis can be specified in respective geoms.
ggplot(diamonds, aes(x=carat, y=price))  # if both X and Y axes are fixed for all layers.
ggplot(diamonds, aes(x=carat, color=cut))  # Each category of the 'cut' variable will now have a distinct  color, once a geom is added.
ggplot(diamonds, aes(x=carat), color="steelblue")
ggplot(diamonds, aes(x=carat, y=price, color=cut)) + geom_point() + geom_smooth()
ggplot(diamonds) + geom_point(aes(x=carat, y=price, color=cut)) + geom_smooth(aes(x=carat, y=price, color=cut)) # Same as above but specifying the aesthetics inside the geoms.
ggplot(diamonds) + geom_point(aes(x=carat, y=price, color=cut)) + geom_smooth(aes(x=carat, y=price)) # Remove color from geom_smooth
ggplot(diamonds, aes(x=carat, y=price)) + geom_point(aes(color=cut)) + geom_smooth()  # same but simpler
ggplot(diamonds, aes(x=carat, y=price, color=cut, shape=color)) + geom_point()

# Approach 1:
data(economics, package="ggplot2")  # init data
economics <- data.frame(economics)  # convert to dataframe
ggplot(economics) + geom_line(aes(x=date, y=pce, color="pcs")) + geom_line(aes(x=date, y=unemploy, col="unemploy")) + scale_color_discrete(name="Legend") + labs(title="Economics") # plot multiple time series using 'geom_line's