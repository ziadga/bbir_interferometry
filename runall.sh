rm *.png
find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {} || echo {} \;
git add .
git commit -am "data run"
git push
