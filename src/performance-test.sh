#!/usr/bin/env bash
# . runscript 80 8 3 stw
size="$1"
eigenvalues="$2"
accuracy="$3"
submethod="$4"

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
  sed -i "" "s/^verbose .*/verbose = -1/" $FILE

  if [[ -n $size ]]; then
    sed -i "" "s/^N .*/N = $1/" $FILE
  else
    echo "Enter the size of problem"
  fi

  if [[ -n $eigenvalues ]]; then
    sed -i "" "s/^M .*/M = $2/" $FILE
  else
    echo "Enter the number of eigenvalues"
  fi

  if [[ -n $submethod ]]; then
    sed -i "" "s/^submethod .*/submethod = $4/" $FILE
  else
    echo "Enter either ftw or stw"
  fi
else
  sed -i "s/^verbose .*/verbose = -1/" $FILE

  if [[ -n $size ]]; then
    sed -i "s/^N .*/N = $1/" $FILE
  else
    echo "Enter the size of problem"
  fi

  if [[ -n $eigenvalues ]]; then
    sed -i "s/^M .*/M = $2/" $FILE
  else
    echo "Enter the number of eigenvalues"
  fi

  if [[ -n $submethod ]]; then
    sed -i "s/^submethod .*/submethod = $4/" $FILE
  else
    echo "Select ftw for ftworth, stw for Stewart, or arpack for Arpack"
  fi
fi

make dune-eigensolver

N=$((size * size))
echo "N = $N"
THREADS="1 4 8 16 32 64 128"
EXP="1 2 3 4"
SIG="1 2"
for threads in $THREADS; do
  for val in $EXP; do
    for x in $SIG; do
    if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i "" "s/^accurate.*/accurate = "$3"/" $FILE
      sed -i "" "s/^tol.*/tol = "$x"e-"$val"/" $FILE
      sed -i "" "s/^numthreads .*/numthreads = $threads/" $FILE
    else
      sed -i "s/^accurate.*/accurate = "$3"/" $FILE
      sed -i "s/^tol.*/tol = "$x"e-"$val"/" $FILE
      sed -i "s/^numthreads .*/numthreads = $threads/" $FILE
    fi
      echo "threads: "$threads"  accuracy: "$3" "$x"e-"$val" $1 $2 $4"
      LOOPPATH=$SRCPATH/adapt_0_symmetric_acc_1_gen_time/N_"$N"_m_"$2"_threads_"$threads"_tol_"$x"e-"$val"_"$4"
      [ -f $LOOPPATH ] && rm -f $LOOPPATH
      ./dune-eigensolver &> $LOOPPATH
    done
  done
done
cd -
