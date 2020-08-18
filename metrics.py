import numpy as np

#Create confusion matrix and evaluating the recall for each class
conv_fun = lambda x : list(x).index(1)

def get_highest(arr):
    temp = np.zeros_like(arr)
    temp[np.arange(len(arr)), arr.argmax(1)] = 1
    return temp


def evaluate_acc_loss_on_devset(test_x,test_y):
    #Evaluate loss 
    devset_loss  = sess.run(loss,feed_dict={reviews:test_x,stars:test_y})
    #Evaluate Test Acc 
    preds   = sess.run(downhill,feed_dict={reviews:test_x,stars:test_y})
    avg_rec = confusion(preds,test_y)
    print('Test Acc :',avg_rec,' %')
    return devset_loss,avg_rec
    
def confusion(preds,gt):
    confusion_mat = np.zeros((preds.shape[1],preds.shape[1]))
    
    #Setting up predictions and gt
    preds = [conv_fun(x) for x in get_highest(preds)]
    gt    = [conv_fun(y) for y in gt]
    
    #Filling confusion matrix
    np.add.at(confusion_mat, (preds, gt), 1)
    #Calculate recall --? why recall Ans: cause I want number of correctly predicted over the actual -GT
    ground_t = np.sum(confusion_mat,axis=0) # GT for all classes
    true_p   = np.diagonal(confusion_mat) #TP predictions for each class

    #It's very unlikely to happen in case of big chunk of data but okay I will do it 
    average_recall = np.nanmean(true_p/ground_t)*100.0
    
    return average_recall