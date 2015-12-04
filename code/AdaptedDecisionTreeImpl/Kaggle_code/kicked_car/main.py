#!/usr/bin/env python

import utilities
import preprocess
import feature_extraction
import csv
import math
import sys

def NumberToWords(number):    
    if (number == 0):
        return "zero"
    if (number < 0):
        return "minus " + NumberToWords(math.fabs(number));

    words = "";
    if ((number /1000000) > 0):
        words += NumberToWords(number / 1000000) + " million "
        number %= 1000000;

    if ((number / 1000) > 0):
        words += NumberToWords(number / 1000) + " thousand "
        number %= 1000;

    if ((number / 100) > 0):
        words += NumberToWords(number / 100) + " hundred "
        number %= 100

    if number > 0:
        if words != "":
            words += "and "

        unitsMap = [ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" ]
        tensMap = [ "zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" ]

        if (number < 20):
            words += unitsMap[number]
        else:
            words += tensMap[number / 10]
            if ((number % 10) > 0):
                words += "-" + unitsMap[number % 10];
        
    return words

def main(training_file, test_file, ratio):
    
    data = utilities.read_file(training_file)
    test_data = utilities.read_file(test_file)

    print 'Preparing data...'
    x, y = preprocess.prepare_data(data)
    refid, x_test = preprocess.prepare_test_data(test_data)
    x, x_test = preprocess.preprocess_features(x, x_test)
    
    print 'Feature extracting...'
    x, x_test = feature_extraction.create_feature(x, y, x_test)
    
    dat = []
    fnum = range(len(x[0]))
    headers = ['class']
    for f in fnum:
        headers.append(str(f))
    dat = [headers]

    for i in range(len(x)):
        c = [y[i]] + x[i]
        dat.append(c)
    
    with open('test_full.csv', 'w') as fp1:
        a = csv.writer(fp1, delimiter=',')
        a.writerows(dat)
        
    x_train, y_train = preprocess.down_sample(x, y, ratio)

    datap = [headers]
    for i in range(len(x_train)):
        c = [y_train[i]] + x_train[i]
        datap.append(c)
    
    print headers;
    with open('train_gen.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(datap)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],1.5)
