# Guidelines To Run Project

## Install dependencies
```
pip install -r requirements.txt
```

### if you want to extract the cold and hinge features you need to:

- create folders Female and put inside the cmp Females pics
- create folders Males and put inside the cmp Males pics
- create folders icdarTrainImages and put inside the icdar train pics
- put the pic directly not inisde another folder

![dirtree](screenshots/1.PNG)

## To Run Tunning & predict module

```
python .\2.Cold_Hinge_Approach\main.py
```

## To Run predict module

- First set ENV argument (TESTDIR) & (OUTDIR)

### if you use CMD
for example:

```
set TESTDIR=E:\CMP\NN\project\Project Submission\test
set OUTDIR=E:\CMP\NN\project\Project Submission\out
```
### if you use Powershell

```
$env:TESTDIR='E:\CMP\NN\project\Project Submission\test'
$env:OUTDIR='E:\CMP\NN\project\Project Submission\out'
```

### Then rum predict module

```
python predict.py
```
