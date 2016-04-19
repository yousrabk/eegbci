#!/bin/bash

function echoHeader() {
  local message=$1
  printf "\033[0;34m"
  echo "================================================================================"
  echo "= $message"
  echo "================================================================================"
  printf "\033[0m"
}

# train on subject which is specified
if [[ ! $1 ]]; then
  echo "No subject specified!"
  exit 1
fi

# setup data for training
echoHeader "training subject: $1"
# sh setup.sh $1

# Find the best model
best_checkpoint=$(ls -1 cv | sort | head -1)

echoHeader "Best model: $best_checkpoint"
echo "$i: $best_checkpoint" >> best_losses

# setup data for testing
num_subjects=12
for ((i=1; i <= $num_subjects; i++)); do
	if [ $i -ne $1 ]; then
		echoHeader "testing subject: $i"
		sh setup.sh $i

		echoHeader "Sampling validation set"
		th sample.lua cv/$best_checkpoint $i
	fi

	python python_utils/calc_roc.py $i

done
