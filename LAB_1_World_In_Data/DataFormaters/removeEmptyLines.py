# Found at https://stackoverflow.com/questions/4521426/delete-blank-rows-from-csv

# This removes all rows that are blank in a CSV file
# After we wrote these programs we learned that in Pandas we could use df = df[df['year']==2018] 
# to remove the lines we didn´t need. But since we already had done it this way we continued with it. 
# This also made it easier to handle CSV files because they became smaller.

import pandas as pd
# Reads an input file
df = pd.read_csv('INPUTFILE.csv')
# Creates a new file and writes to it.
df.to_csv('OUTPUTFILE', index=False)