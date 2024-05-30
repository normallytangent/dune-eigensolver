#!/usr/bin/env bash
FILE=dune-eigensolver.ini

make dune-eigensolver

sed -i "" "s/verbose .*/verbose = 0/" $FILE

if [ $1 == "std" ]; then
  sed -i "" "s/method .*/method = std/" $FILE
elif [ $1 == "gen" ]; then
  sed -i "" "s/method .*/method = gen/" $FILE
fi
#  sed regularization

#sed 
#sed tol
#sed  
#
#for ; do 
#  sed 
#  sed
#  sed
#  ./eigensolver &> $1
#done
