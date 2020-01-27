#!/bin/sh

$USAGE = "./db_util.sh  [-c ]"

# TODO input validation
# execute program based on options
while [ -n "$1" ]; do
	case "$1" in
		-c) 
			./createDB.sh $2
			;;
		-s)
			./copyDB.sh $2 $3
			;;
		-e) python export.py $2 $3 $4
			;;
		-t) python craeteTrain.py $2 $3 $4
			;;
		-h) echo $USAGE ;;
	esac
done