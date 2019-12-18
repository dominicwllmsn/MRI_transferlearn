while read line
do 
find ./ADNI_comb -name "$line" | xargs cp -t ./unfinishedNii88/
done < difference_comb.txt
