rm *.png
find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec echo {} \;
#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ \;
find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ --loadref --refname out12.txt \;
git add .
git commit -am "data run"
git push
