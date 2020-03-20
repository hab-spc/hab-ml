#!/bin/bash
# run this script to create the barebones database

sqlite3 $1 <<EOF
create table images (Filename TEXT PRIMARY KEY,GTruth INTEGER,DateCollected TEXT,Prediction INTEGER,Orientation REAL,majorLength REAL,minorLength REAL,Confidence REAL,Height REAL,Width REAL);
EOF
