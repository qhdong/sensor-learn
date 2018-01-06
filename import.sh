#!/bin/sh

FILENAME=sensor-`date +%F`.tar.gz
scp titan:/home/ubuntu/Playground/NodeJS/sensor-data/one-digit/$FILENAME ./data/

cd data
tar zxvf $FILENAME
mongorestore -d sensor sensor

