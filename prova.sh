#!/bin/bash

echo 'sentence 201806130752-9992065 layer 1 word lesioni' >> /home/fusco/bt/cd/out/201806130752-9992065/lesioni_1.txt
for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
  echo 'head' $i  >> /home/fusco/bt/cd/out/201806130752-9992065/lesioni_1.txt
  echo 'head' $i  
  python liBERTy.py see_token -l1 -h $i -w lesioni -s -t 5 >> /home/fusco/bt/cd/out/201806130752-9992065/lesioni_1.txt
  echo '---------------------------------------------------------' >> /home/fusco/bt/cd/out/201806130752-9992065/lesioni_1.txt
done
  
