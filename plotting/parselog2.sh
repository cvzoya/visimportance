#!/bin/bash
# Usage parselog.sh caffe.log 
# It creates two files one caffe.log.test that contains the loss and test accuracy of the test and
# another one caffe.log.loss that contains the loss computed during the training

if [ "$#" -lt 1 ]
then
echo "Usage parselog.sh /path/to/your.log"
exit
fi

LOG=`basename $1`

# -------------- TESTING LOG --------------

# find this phrase and copy the line with this phrase and the line above to aux.txt
grep -B 1 'Test net output ' $1 > aux.txt

# search the file aux.txt and extract all the iteration numbers to aux0.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt

# copy the test net loss to aux1.txt
#grep ': loss = ' aux.txt > aux1.txt
#grep 'loss = ' aux1.txt | awk '{print $9}' > aux2.txt
#grep -E -o 'loss =.{0,9}' aux.txt > aux1.txt
grep 'loss = ' aux.txt | sed 's/.*loss = \([[:graph:]]*\).*/\1/g' > aux1.txt

# take everything after the loss
#cut -c8-14 aux1.txt > aux2.txt
#cut -d $'\n' -f8- aux1.txt > aux2.txt

# run this if you want a header in your file
#echo '# Iters TestingLoss'> $LOG.test

# merge the iteration number and test loss in test log
paste aux0.txt aux1.txt | column -t >> $LOG.test

# remove auxiliary files
rm aux0.txt aux1.txt


# -------------- TRAINING LOG --------------

## extract all lines that have loss on them (to aux.txt), and save the iteration number (to aux0.txt)
grep ', loss = ' $1 > aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt

## extract all loss values to axu1.txt (note test loss is preceded by a colon, so won't be picked up)
grep ', loss = ' $1 | awk '{print $9}' > aux1.txt

## get all learning rates
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt

## run this if you want a header in your file
##echo '# Iters TrainingLoss LearningRate'> $LOG.train

## merge itereation number, loss values, and learning rates as columns in train log
paste aux0.txt aux1.txt aux2.txt | column -t >> $LOG.train

## remove auxiliary files
rm aux.txt aux0.txt aux1.txt aux2.txt
