{
read
while IFS=, read -r N T G S; do
    venv/bin/python src/newdist.py "$N" "$T" "$G" "$S"
done
} < sim/params1.csv
