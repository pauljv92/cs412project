import csv
import sys

def loadandtrain(filename):
    linenum = 0    
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    
    with open(filename) as fp:
        for line in fp:
            linenum += 1
            if(line.strip() == ''):
                continue;
            tokens = line.split();
            classcount[tokens[0]] += 1
            for token in tokens[1:]:
                if token in labelcount[tokens[0]]:
                    labelcount[tokens[0]][token] += 1
                else:
                    labelcount[tokens[0]][token] = 1
                    
    return linenum, labelcount, classcount

def test(filename,labelcnt,classcnt):
    linenumt = 0
    correctt = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with open(filename) as fpt:
        for line in fpt:
            linenumt += 1 
            if(line.strip()==''):
                continue
            tokenst = line.split();
            classval = tokenst[0]
            predclass = argmax(labelcnt,tokenst[1:],classcnt)
            if(classval == predclass):
                correctt += 1
                if(classval == "+1"):
                    tp +=1
                else:
                    tn +=1
            else:
                if(classval == "-1"):
                    fn +=1
                else:
                    fp +=1
    
    #print "The total number is"+str(linenumt)
    #print "The total correct is"+str(correctt)
    #print "The accuracy is "+str(correctt/(float(linenumt)))
    print str(tp)+" "+str(fn)+" "+str(fp)+" "+str(tn)
    return correctt/(float(linenumt)), tp, fn, fp, tn
    

def condprob(lblcnt, rndvar, classval,classcnt):
    return lblcnt[classval][rndvar]/float(classcnt[classval])
    
def argmax(lblcnt, varbs,classcnt):
    neg = 1.0
    pos = 1.0
    #varbs = datapoint.split();
    for var in varbs:
        if var not in lblcnt["-1"].keys():
            neg = neg*0.0000001;
        else:
            neg = neg*((lblcnt["-1"][var])/float(classcnt["-1"]))

        if var not in lblcnt["+1"].keys():
            #smooth by adding small probability
            pos = pos*0.0000001;
        else:
            #smooth by adding small probability
            pos = pos*((lblcnt["+1"][var])/float(classcnt["+1"]))
            
    return "-1" if (neg > pos) else "+1"

def NBrun(trainfile,testfile):
    linenum = 0
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    
    linenum, labelcount, classcount = loadandtrain(trainfile);

    return test(testfile,labelcount,classcount);
    #test(testfile,labelcount,classcount);

if __name__ == "__main__":    
    
    linenum = 0
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}

    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    #print trainfile
    #print testfile
    linenum, labelcount, classcount = loadandtrain(trainfile);

    test(trainfile,labelcount,classcount);
    test(testfile,labelcount,classcount);

    
#for i in labelcount["+1"]:
#   print "The count for "+str(i)+" is: "+str(labelcount["+1"][i])
    
#print condprob(labelcount,"73:1","+1",classcount)
#total = classcount["+1"] + classcount["-1"]
#print classcount["+1"]
#print classcount["-1"]
#print total

    
