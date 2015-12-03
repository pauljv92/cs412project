import sys
import random
import math
from copy import deepcopy

def load(filename):
    linenum = 0
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    pointw = []
    
    with open(filename) as fp:
        for line in fp:
            linenum += 1
            if(line.strip()==''):
                linenum-=1
                continue
            tokens = line.split();
            classcount[tokens[0]] += 1
            pointw.append([''.join(line),1.0]);
            for token in tokens[1:]:
                if token in labelcount[tokens[0]]:
                    labelcount[tokens[0]][token] += 1
                else:
                    labelcount[tokens[0]][token] = 1

    for i in pointw:
        i[1] /= linenum 
                    
    return linenum, labelcount, pointw, classcount
    
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

def trainNB(labelcount,classcount,filename):
    linenumt = 0
    correctt = 0
    
    with open(filename) as fpt:
        for line in fpt:
            linenumt += 1
            if(line.strip()==''):
                linenumt-=1
                continue
            tokenst = line.split();
            classval = tokenst[0]
            predclass = argmax(labelcount,tokenst[1:],classcount)
            
            if(classval == predclass):
                correctt += 1
            
    #print "The accuracy is "+str(correctt/(float(linenumt)))
    return str(correctt/float(linenumt))

def trainAda(pointw, labelcount,classcount):
    D_total = classcount["-1"] + classcount["+1"]
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0,"-1":0}
    
    weighted_probs = []
    for i in pointw:
        if(len(weighted_probs)==0):
            weighted_probs.append(i[1])
        else:
            weighted_probs.append(weighted_probs[-1]+i[1])
            
    sampleindx = {}
    for j in range(len(pointw)):
        rnd = random.random()
        k = 0;
        while(rnd > (weighted_probs[k])):
            k +=1

        if k in sampleindx:
            sampleindx[k] += 1
        else:
            sampleindx[k] = 1

    i = 0;
    for s in sampleindx:
        #print pointw[s][0][:2]
        val = pointw[s][0]
        tokens = val.split()
        classcount[tokens[0]] += sampleindx[s]
        for token in tokens[1:]:
            if token in labelcount[pointw[s][0][:2]]:
                labelcount[pointw[s][0][:2]][token] += sampleindx[s]
            else:
                labelcount[pointw[s][0][:2]][token] = sampleindx[s]
              
    return labelcount, classcount

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
            neg = neg*condprob(lblcnt,var,"-1",classcnt)

        if var not in lblcnt["+1"].keys():
            #smooth by adding small probability
            pos = pos*0.0000001;
        else:
            #smooth by adding small probability
            pos = pos*condprob(lblcnt,var,"+1",classcnt)
            
    return "-1" if (neg > pos) else "+1"

def test(filename,labelcnt,classcnt,pointw):
    linenumt = 0
    correctt = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with open(str(filename)) as fpt:
        for line in fpt:
            linenumt += 1 
            if(line.strip()==''):
                linenumt-=1
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
    
    #print str(tp)+" "+str(fn)+" "+str(fp)+" "+str(tn)
    accuracy = correctt/(float(linenumt))
    #print("accuracy:", accuracy)
    alpha =  0.5*math.log(accuracy/(1-accuracy))
  
    linenumt = -1
    newpointw = deepcopy(pointw)

    with open(str(filename)) as fpt:
        for line in fpt:
            linenumt += 1
            if(line.strip()==''):
                linenumt-=1
                continue
            tokenst = line.split();
            classval = tokenst[0]
            predclass = argmax(labelcnt,tokenst[1:],classcnt)
            if(classval == predclass):
                newpointw[linenumt][1] /= math.exp(alpha);
            else:
                newpointw[linenumt][1] *= math.exp(alpha);

    return alpha, newpointw
    
def normalizepointw(pointw):
    totalpw = 0.0;
    checksum = 0.0;

    for i in range(len(pointw)):
        totalpw += pointw[i][1]
    
    for i in range(len(pointw)):
        pointw[i][1] /= totalpw
        checksum += pointw[i][1]

    #print "The sum is "+str(checksum)

def testAda(classifiers,filename,k):
    linenumt = 0
    correctt = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with open(str(filename)) as fpt:
        for line in fpt:
            linenumt += 1 
            tokenst = line.split();
            if len(tokenst)==0:
                continue
            classval = tokenst[0]
            predclass = 0.0;
            for i in range(k):
                alpha, pointw, labelcount, classcount = classifiers[i]
                predclass += alpha * float(int(argmax(labelcount,tokenst[1:],classcount)))
                
            if(predclass > 0.0):
                predclass = "+1"
            else:
                predclass = "-1"

            if(classval == predclass):
                correctt += 1
                if(classval == "+1"):
                    tp += 1
                else:
                    tn += 1
            else:
                if(classval == "-1"):
                    fn += 1
                else:
                    fp += 1
                    
    #print "The total number is"+str(linenumt)
    #print "The total correct is"+str(correctt)
    #print "The accuracy is "+str(correctt/(float(linenumt)))
    print str(tp)+" "+str(fn)+" "+str(fp)+" "+str(tn)
    acc = correctt/(float(linenumt))
    return (acc, tp, fn, fp, tn)

def NBAdarun(trainfile, testfile,k):
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    classifiers = []
    
    linenum, labelcount, pointw, classcount = load(trainfile);

    for i in range(k):        
        labelcount, classcount = trainAda(pointw,labelcount,classcount)
        alpha, newpointw = test(trainfile,labelcount,classcount,pointw);
        classifiers.append((alpha,pointw,labelcount,classcount))
        pointw = newpointw
        normalizepointw(pointw);

    return testAda(classifiers,trainfile,k) + testAda(classifiers,testfile,k)
    
if __name__ == "__main__":    
    
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    classifiers = []
    k = 8
    
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    #print trainfile
    #print testfile
    linenum, labelcount, pointw, classcount = load(trainfile);
    
    for i in range(k):        
        labelcount, classcount = trainAda(pointw,labelcount,classcount)
        #print labelcount
        #print classcount
        alpha, newpointw = test(trainfile,labelcount,classcount,pointw);
        classifiers.append((alpha,pointw,labelcount,classcount))
        pointw = newpointw
        #print alpha
        normalizepointw(pointw);
        #print pointw

    testAda(classifiers,testfile,k)
    testAda(classifiers,trainfile,k)

    #trainNB(labelcount,classcount,testfile);
    #print pointw
    #test(trainfile,labelcount,classcount);
    #test(testfile,labelcount,classcount);
