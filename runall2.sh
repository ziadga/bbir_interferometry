#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec echo {} \;
#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ \;
#mkdir 2020-12-15
#mv *.png *.txt 2020-12-15

#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-16/ -mindepth 1 -maxdepth 1 -type d -exec echo {} \;
#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-16/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ \;
#mkdir 2020-12-16
#mv *.png *.txt 2020-12-16

#find /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ --loadref --refname out4.txt --outname ref4_ \;
#mv *.png *.txt 2020-12-15

find /gpfs/scratch60/ganim/zg54/aerys/2020-12-16/ -mindepth 1 -maxdepth 1 -type d -exec python interf_plot.py --dir {}/ --loadref --refname out8.txt --outname ref8_ \;
#mv *.png *.txt 2020-12-16

#python interf_plot.py --dir /gpfs/scratch60/ganim/zg54/aerys/2020-12-15/test06-S4-QDs\ check\ scan\ range/ --loadref --refname out18.txt --outname ref18_
