## Guidelines

- Put Dataset of CMP23 as downloaded in the same directory you clone this repo to
- Put Dataset of ICDAR as below
- ![datasetICDAR](screenshots/1.jpg)
- Train data contains images from 0001_* to 0283_*
- Test data contains images from 284\_\* till end
- You must delete arabic samples in ICDAR dataset, you can do it simply in cmd opend at images_gender/images/train

```
del .*_1.jpg
del .*_2.jpg
```

- if you want to extract the cold and hinge features you need to:
- create folders Female and put inside the cmp Females pics
- create folders Males and put inside the cmp Males pics
- create folders icdarTrainImages and put inside the icdar train pics
- put the pic directly not inisde another folder
-![](screenshots/1.PNG)

