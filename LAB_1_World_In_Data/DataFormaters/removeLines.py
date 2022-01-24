import csv 

# Found at https://stackoverflow.com/questions/29725932/deleting-rows-with-python-in-a-csv-file
# After we wrote these programs we learned that in Pandas we could use df = df[df['year']==2018] 
# to remove the lines we didn´t need. But since we already had done it this way we continued with it. 
# This also made it easier to handle CSV files because they became smaller.
# If a row at index 2 doesnt have 2018 we don't rewrite it.
with open('OUTPUTFILE', 'w+') as output_file:
    with open('INPUTFILE') as input_file: 
      writer = csv.writer(output_file)
      for row in csv.reader(input_file):
        if row[2] == "2018":
          writer.writerow(row)




