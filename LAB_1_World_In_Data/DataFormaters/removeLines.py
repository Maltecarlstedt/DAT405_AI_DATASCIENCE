import csv 

# Found at https://stackoverflow.com/questions/29725932/deleting-rows-with-python-in-a-csv-file

# If a row at index 2 doesnt have 2018 we don't rewrite it.

with open('OUTPUTFILE', 'w+') as output_file:
    with open('INPUTFILE') as input_file: 
      writer = csv.writer(output_file)
      for row in csv.reader(input_file):
        if row[2] == "2018":
          writer.writerow(row)




