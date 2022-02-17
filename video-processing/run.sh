#!/bin/bash
../../Scad/runtime/lib/memory_server "2333">&/dev/null &
../../Scad/runtime/lib/memory_server "2334">&/dev/null &
../../Scad/runtime/lib/memory_server "2335">&/dev/null &
../../Scad/runtime/lib/memory_server "2336">&/dev/null &

rundisagg func1.o.py -m "mem1:2333" "mem2:2334" "mem3:2335" "mem4:2336">&/dev/null 
rundisagg func2.o.py -m "mem1:2333" "mem2:2334" "mem3:2335" "mem4:2336">&/dev/null 
rundisagg func3.o.py -m "mem1:2333" "mem2:2334" "mem3:2335" "mem4:2336">&/dev/null

kill -9 $(lsof -i:2333 -t)
kill -9 $(lsof -i:2334 -t)
kill -9 $(lsof -i:2335 -t)
kill -9 $(lsof -i:2336 -t)