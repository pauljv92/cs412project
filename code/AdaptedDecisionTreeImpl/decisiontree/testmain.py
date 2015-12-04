import NaiveBayes as NB
import NBAdaboost as NBA
import sys

if __name__ == "__main__":    
    
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    
    maxtrainacc = 0.0;
    maxtestacc = 0.0;
    maxacc_kval = 0;
    optmets_train = [0,0,0,0]
    optmets_test = [0,0,0,0]

    for k in range(3,30):
        if(k%3==0):
            print "The current k is "+str(k)
            NBAmets = NBA.NBAdarun(trainfile,testfile,k)
            if NBAmets[0] > maxtrainacc:
                maxacc_kval = k;
                maxtrainacc = NBAmets[0];
                optmets_train[0] = NBAmets[1]
                optmets_train[1] = NBAmets[2]
                optmets_train[2] = NBAmets[3]
                optmets_train[3] = NBAmets[4]

                maxtestacc = NBAmets[5]
                optmets_test[0] = NBAmets[6]
                optmets_test[1] = NBAmets[7]
                optmets_test[2] = NBAmets[8]
                optmets_test[3] = NBAmets[9]             
    NBmets = NB.NBrun(trainfile,testfile)
    
    print "The optimal size classifier would have "+str(maxtrainacc)+"% accuracy on the training set "+str(maxtestacc)+"% accuracy on the testing set and a k-value of"+str(maxacc_kval)

    print "TP: "+str(optmets_test[0])
    print "FN: "+str(optmets_test[1])
    print "FP: "+str(optmets_test[2])
    print "TN: "+str(optmets_test[3])

    print "The Naive Bayes accuracy is: "+str(NBmets[0])+"% on the testing set."
    print "TP: "+str(NBmets[1])
    print "FN: "+str(NBmets[2])
    print "FP: "+str(NBmets[3])
    print "TN: "+str(NBmets[4])




