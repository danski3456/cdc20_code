echo "Number of graphs to compare in parallel. To test it try 'simple', else 'complete'"
read param
cat "sim/params_$param.csv" | parallel -C, --header : "venv/bin/python sim/sim_compare_graphs.py {N} {T} {G} {S}" &
