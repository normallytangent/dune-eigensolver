#!/usr/bin/env bash
# . runscript 8 stw 0
eigenvalues="$1"
submethod="$2"
adapt="$3"

if [[ "$OSTYPE" == "darwin"* ]]; then
  SRCPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/dune-eigensolver/src
  BINPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/release-build/dune-eigensolver/src
else
  SRCPATH=/home/svaishnavi/TRYIT/DUNE/dune-eigensolver/src
  BINPATH=/home/svaishnavi/TRYIT/DUNE/release-build/dune-eigensolver/src
fi

FILE=$SRCPATH/dune-eigensolver.ini

cd $BINPATH

if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i "" "s/^verbose .*/verbose = 0/" $FILE

  if [[ -n $eigenvalues ]]; then
    sed -i "" "s/^M .*/M = $1/" $FILE
  else
    echo "Enter the number of eigenvalues"
  fi

  if [[ -n $submethod ]]; then
    sed -i "" "s/^submethod .*/submethod = $2/" $FILE
  else
    echo "Enter either GeneralizedInverse:ftw or GeneralizedSymmetricStewart:stw"
  fi
  if [[ -n $adapt ]]; then
    sed -i "" "s/^adapt .*/adapt = $3/" $FILE
  else
    echo "Enter either off:0 or on:1"
  fi
else
  sed -i "s/^verbose .*/verbose = 0/" $FILE

  if [[ -n $eigenvalues ]]; then
    sed -i "s/^M .*/M = $1/" $FILE
  else
    echo "Enter the number of eigenvalues"
  fi

  if [[ -n $submethod ]]; then
    sed -i "s/^submethod .*/submethod = $2/" $FILE
  else
    echo "Enter either GeneralizedInverse:ftw or GeneralizedSymmetricStewart:stw"
  fi
  if [[ -n $adapt ]]; then
    sed -i "s/^adapt .*/adapt = $3/" $FILE
  else
    echo "Enter either off:0 or on:1"
  fi
fi

make dune-eigensolver
SUBDOMAIN="1 631 2604 7722 11819 13590"
ACC="1 2 3 4"
EXP="1 2 3 4"
SIG="1 2"
for sub in $SUBDOMAIN; do 
  for acc in $ACC; do
    for val in $EXP; do
      for x in $SIG; do
      if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i "" "s/^subdomain.*/subdomain = "$sub"/" $FILE
        sed -i "" "s/^accurate.*/accurate = "$acc"/" $FILE
        sed -i "" "s/^tol.*/tol = "$x"e-"$val"/" $FILE
      else
        sed -i "s/^subdomain.*/subdomain = "$sub"/" $FILE
        sed -i "s/^accurate.*/accurate = "$acc"/" $FILE
        sed -i "s/^tol.*/tol = "$x"e-"$val"/" $FILE
      fi
        echo "$sub" "accuracy: "$acc" "$x"e-"$val" $1 $2 $3"
        LOOPPATH=$SRCPATH/checkerboard-data/"$2"_adapt_"$3"_domain_"$sub"_m_"$1"_acc_"$acc"_tol_"$x"e-"$val"
        [ -f $LOOPPATH ] && rm -f $LOOPPATH
        ./dune-eigensolver &> $LOOPPATH
      done
    done
  done
done
cd -
