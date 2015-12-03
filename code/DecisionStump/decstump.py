import numpy
import NBAdaboost as NBA
import sys
import math

def conv2binaryf(pointw,classes):
    labels = []
    data = []

indexes = range(1,2)

    for i in pointw:
        tokens = i[0].split(',');
        labels.append(tokens[0])
        data.append(tokens[1:])
        #data2.append(tokens[1:])

    for x in data:
        for indx in indexes:
            x[indx] = float(x[indx])
    
    for indx in indexes:
        sorted_data = [y[indx] for y in sorted(data, key=lambda z: z[indx])]
        size = int(math.sqrt(len(sorted_data)))

        bins = [] #(min, max, centroid)
        acc = 0
        while sorted_data:
            chunk = sorted_data[:size]
            bins.append((min(chunk), max(chunk), sum(chunk)/len(chunk)))
            acc += len(chunk)
            sorted_data = sorted_data[size:]

        for x in data:
            # find the right bin, and replace with centroid
            for a, b, c in bins:
                if x[indx] >= a and x[indx] <= b:
                    x[indx] = c
                    break

        #elems = [x[indx] for x in data]
        #print "Elements length =", len(set(elems))
        
    '''
    print len(labels)
    allvals = {}
   
    classes = classes.split(',')
    classes = classes[1:]
    
    for i in range(len(classes)):
        allvals[classes[i]] = []
        for k in range(len(data)):
            if data[k][i] not in allvals[classes[i]]:
                allvals[classes[i]].append(data[k][i])
    
    print allvals
    
    datasize = len(data)
    numstumps = 0
    for c in range(len(classes)):
        varbs = allvals[classes[c]] #all possibles values of class -> classes[c]
        for v in varbs:
            print "For class: "+str(classes[c])+" = "+str(v)
            ones = 0;
            totalv = 0;
            zeros = 0;
            totale = 0;
            for i in range(datasize):
                #We are testing for hypothesis >= var is +
                #print "Point - "+str(i)
                #print float(data[i][c]), float(v)
                if float(data[i][c]) >= float(v):
                    totalv += 1
                    if int(labels[i]) == 1:
                        #print "+ label datapoint"
                        ones += 1
                else:
                    totale += 1
                    if int(labels[i]) == 0:
                        #print " - label datapoint"
                        zeros += 1

            #print ones,totale,totalv
            accuracy = ones/float(totale+totalv)
            #print "One are "+str(ones)
            #print "The accuracy is:" +str(accuracy)
            if(accuracy < 0.5):
                continue;
            else:
                numstumps += 1
                print "For class: "+str(classes[c])+" = "+str(v)+", the accuracy is: "+str(ones/float(totale+totalv))
                
   '''
    
def choosestump(pointw,classes):
    labels = []
    data = []
    for i in pointw:
        tokens = i[0].split(',');
        labels.append(tokens[0])
        data.append(tokens[1:])

    #print labels
    #print classes
    #print pointw
    print len(labels)
    allvals = {}
   
    classes = classes.split(',')
    classes = classes[1:]
    #print classes

    for i in range(len(classes)):
        allvals[classes[i]] = []
        for k in range(len(data)):
            if data[k][i] not in allvals[classes[i]]:
                allvals[classes[i]].append(data[k][i])
    #print allvals
    
    datasize = len(data)
    numstumps = 0
    for c in range(len(classes)):
        varbs = allvals[classes[c]] #all possibles values of class -> classes[c] 
        for v in varbs:
            ones = 0;
            totalv = 0;
            zeros = 0;
            totale = 0;
            for i in range(datasize):
                #When we are
                #print "Point - "+str(i)
                #print float(data[i][c]), float(v)
                if float(data[i][c]) >= float(v):
                    totalv += 1
                    if int(labels[c]) == 1: 
                        ones += 1
                else:
                    totale += 1
                    if int(labels[c]) == 0:
                        zeros += 1

            accuracy = ones/float(totale+totalv)
            #print accuracy
            if(accuracy < 0.5):
                continue;
            else:
                numstumps += 1
                print "For class: "+str(classes[c])+" = "+str(v)+", the accuracy is: "+str(ones/float(totale+totalv))
                
        
        #print "The labels are:"+str(labels)
        
    print "The number of stumps is "+str(numstumps)
        
if __name__ == "__main__":    
    
    labelcount = {"+1":{}, "-1":{}}
    classcount = {"+1":0, "-1":0}
    classifiers = []
    k = 500
    
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    
    linenum,pointw,classes = NBA.load(trainfile)
    
    #print classes
    #choosestump(pointw,classes)
    conv2binaryf(pointw,classes)

'''    
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


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    
    dataMatrix = mat(dataArr); 
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf     
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']: 
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
'''
