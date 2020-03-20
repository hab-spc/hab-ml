#!/bin/sh
# run this script to copy over the local db to svcl db
db_path = $1
serverDB_path = $2

scp $db_path $serverDB_path