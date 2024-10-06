Sections:
* Dataset


** Dataset
This HIV Dataset from
The results of the screening test released in May 2004 placed each compound in one of three categories.

CA - Confirmed Active
CM - Confirmed Moderately Active
CI - Confirmed Inactive

In this investigation I will be creating a model to classify molecules based on whether they are confirmed HIV active or not.
Consequently, classes CA and CM will be merged into one and are assigned the label 1. CI will be assigned the label 0.

One difficulty with this dataset revolves around the number of HIV positive samples.
There are 1443 HIV positive molecules and 39684 HIV negative molecules.

Test set results: loss= 0.4025 accuracy= 0.8580 recall= 0.4158 precision= 0.1084 f1= 0.1720
