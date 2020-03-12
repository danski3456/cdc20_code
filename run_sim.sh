cat sim/params1.csv | parallel -C, --header : "venv/bin/python sim/sim_compare_graphs.py {N} {T} {G} {S}" &
