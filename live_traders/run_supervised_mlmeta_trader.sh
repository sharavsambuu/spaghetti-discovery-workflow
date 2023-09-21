#!/bin/bash

source ../env/bin/activate
until python mlmeta_live_trader.py
do
    echo "Restarting"
    sleep 5
done
