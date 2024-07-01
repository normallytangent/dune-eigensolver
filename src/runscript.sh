#!/usr/bin/env bash
# . runscript std stw dir

method="$1"
submethod="$2"
matrix="$3"
SRCPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/dune-eigensolver/src
BINPATH=/Users/vsanc/Documents/1UHD/7-wise-22-23/TRYIT/DUNE/release-build/dune-eigensolver/src
FILE=$SRCPATH/dune-eigensolver.ini

cd $BINPATH
sed -i "" "s/^verbose .*/verbose = 1/" $FILE

if [[ -n $method ]]; then
  sed -i "" "s/^method .*/method = $1/" $FILE
else
  echo "Enter either std or gen"
fi

if [[ -n $submethod ]]; then
  sed -i "" "s/^submethod .*/submethod = $2/" $FILE
else
  echo "Enter either ftw or stw"
fi

make dune-eigensolver

EXP="1 2 3 4"
SIG="5 6 7 8 9"
for val in $EXP; do
  for x in $SIG; do
    sed -i "" "s/^tol.*/tol = "$x"e-"$val"/" $FILE
    echo ""$x"e-"$val" $1 $2 $3"
    ./dune-eigensolver &> $SRCPATH/measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_"$x"e-"$val"_overlap_3_method_"$1"_submethod_"$2"_"$3"
  done
done
cd -
