def csv_reader(file, header=True):
    X_train = []
    y_train = []
    train_coutn = 0

    X_foreign = []
    ext_count = 0
    with open(file, newline='') as csvfile:
        readCSV = csv.reader(csvfile)#, delimiter=',')
        for line in readCSV: #Iterate through the loop to read line by line
            if line[-1] == '':
                X_foreign.append(' '.join(re.findall(r"[\w']+",line[0])))
                ext_count += 1
            else:
                X_train.append(' '.join(re.findall(r"[\w']+",line[0])))
                y_train.append(' '.join(re.findall(r"[\w']+",line[-1])))

                train_coutn += 1
                
        csvfile.close()
        
    if header:
        X_train.pop(0)
        y_train.pop(0)
        
    return X_train, y_train, X_foreign