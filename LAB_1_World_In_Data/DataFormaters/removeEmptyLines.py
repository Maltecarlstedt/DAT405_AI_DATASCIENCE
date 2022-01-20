# Found at https://stackoverflow.com/questions/4521426/delete-blank-rows-from-csv

# This removes all rows that are blank in a CSV file

import pandas as pd
df = pd.read_csv('INPUTFILE.csv')
df.to_csv('OUTPUTFILE', index=False)