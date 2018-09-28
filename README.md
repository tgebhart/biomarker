# Biomarker

Machine learning for biomarker detection data.

### Requirements:

python 2
pandas
sklearn
numpy
jupyter

## To run:

clone the repository then run

```
$ pip install -r requirements.txt
```

Also install the directory as a package so you can call functions and stuff
without worrying about directory structure. From the root of the directory:

```
$ pip install -e .
```

The code is currently completely contained in the Jupyter notebook EDA.ipynb. To view/edit this notebook, run

```
$ jupyter notebook
```

This should open a browser tab with the file directory. Navigate to `notebooks/EDA.ipynb` to view/edit the file.

I have also created an HTML copy of the file in the same directory that can be used to view the code without
following the previous steps.

## To Do:

* Run algorithm on easily-matrixed portion of data
* Report results on above analysis
* Determine best way to convert all data into matrix/algorithm-interpretable format
* etc.
