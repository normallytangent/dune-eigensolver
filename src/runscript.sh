#!/usr/bin/env bash
# . runscript 80 8 std stw dir
size="$1"
eigenvalues="$2"
method="$3"
submethod="$4"
matrix="$5"
SRCPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/dune-eigensolver/src
BINPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/release-build/dune-eigensolver/src
FILE=$SRCPATH/dune-eigensolver.ini

cd $BINPATH
sed -i "" "s/^verbose .*/verbose = 0/" $FILE

if [[ -n $size ]]; then
  sed -i "" "s/^N .*/N = $1/" $FILE
else
  echo "Enter the size of problem"
fi

if [[ -n $eigvalues ]]; then
  sed -i "" "s/^M .*/M = $2/" $FILE
else
  echo "Enter the number of eigenvalues"
fi

if [[ -n $method ]]; then
  sed -i "" "s/^method .*/method = $3/" $FILE
else
  echo "Enter either std or gen"
fi

if [[ -n $submethod ]]; then
  sed -i "" "s/^submethod .*/submethod = $4/" $FILE
else
  echo "Enter either ftw or stw"
fi

make dune-eigensolver

N=$((size * size))
echo "N = $N"
ACC="1 2 3 4"
EXP="1 2 3 4"
SIG="1 2"
for acc in $ACC; do
  for val in $EXP; do
    for x in $SIG; do
      sed -i "" "s/^accurate.*/accurate = "$acc"/" $FILE
      sed -i "" "s/^tol.*/tol = "$x"e-"$val"/" $FILE
      echo ""$x"e-"$val" $1 $2 $3 $4 $5"
      LOOPPATH=$SRCPATH/feb25/N_"$N"_m_"$2"_acc_"$acc"_tol_"$x"e-"$val"_"$3"_"$4"_"$5"
      [ -f $LOOPPATH ] && rm -f $LOOPPATH
      ./dune-eigensolver &> $LOOPPATH
    done
  done
done
cd -
