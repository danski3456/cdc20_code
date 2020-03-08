cat sim/params1.csv | parallel -C, --header : "venv/bin/python sim/basic_sim3.py {N} {T} {G} {S}" &
