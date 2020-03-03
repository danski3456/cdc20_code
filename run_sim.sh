cat sim/params1.csv | parallel -C, --header : "venv/bin/python src/newdist.py {N} {T} {G} {S} > sim/{N}_{T}_{G}_{S}.out" &
