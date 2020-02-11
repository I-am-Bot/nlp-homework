import pandas as pd
import numpy as np

rest_train = pd.read_csv("./restauranttrain.bio", encoding = "ISO-8859-1", delim_whitespace = True)
rest_test = pd.read_csv("./restauranttest.bio", encoding = "ISO-8859-1", delim_whitespace = True)

print(rest_train)
print(rest_test)

