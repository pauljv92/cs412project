import math
import sys

def sensitivity(tp,fn):
    return tp/float(tp+fn)

def specificity(tn,fp):
    return tn/float(tn+fp)

def precandrecall(tp,fp,fn):
    prec = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    return prec, recall

def Fbeta(prec,recall,beta):
    betasq = beta*beta
    a = ((1+betasq)*prec*recall)
    b = ((betasq*prec)+recall)
    
    return a/float(b)

if __name__ == "__main__":    
    
    tp = int(sys.argv[1])
    fn = int(sys.argv[2])
    fp = int(sys.argv[3])
    tn = int(sys.argv[4])

    print "Sensitivity: "+str(sensitivity(tp,fn))
    print "Specificity: "+str(specificity(tn,fp))
    prec, recall = precandrecall(tp,fp,fn)    
    print "Precision : "+str(prec)
    print "Recall : "+str(recall)

    beta = [0.5,1,2]
    for b in beta:
        print "FBeta_"+str(b)+" = "+str(Fbeta(prec,recall,b))

