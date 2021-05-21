for entry in `ls ./data`
do
  echo ${entry}
  /usr/local/bin/python3 mdp.py -f ${entry} > results/${entry}
done