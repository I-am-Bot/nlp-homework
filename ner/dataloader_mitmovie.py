import pandas as pd
import numpy as np

movie_train = pd.read_csv("./engtrain.bio", encoding = "ISO-8859-1", delim_whitespace = True)
movie_test = pd.read_csv("./engtest.bio", encoding = "ISO-8859-1", delim_whitespace = True)

print(movie_train)
print(movie_test)

