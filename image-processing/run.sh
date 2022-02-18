#!/bin/bash
../../Scad/runtime/lib/memory_server "2333">&/dev/null &

rundisagg func1.o.py -m "mem1:2333">&/dev/null 
rundisagg func2.o.py -m "mem1:2333">&/dev/null 
rundisagg func3.o.py -m "mem1:2333">&/dev/null

kill -9 $(lsof -i:2333 -t)