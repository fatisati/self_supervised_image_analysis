## infimnist-parser
A python parser that converts infinite MNIST binary data to a readable format with a user-supplied delimiter

#### Example

* First generate binary data with the infimnist executable downloaded from <http://leon.bottou.org/projects/infimnist>:
```
infimnist lab 0 9999 > test10k-labels
infimnist pat 0 9999 > test10k-patterns
```
* Then convert the binary data to a human readable format with any delimiter:
```
python infimnist_parser.py test10k-labels test10k-patterns -d '|' -o test10k
```
* Or use the parser in the interactive mode or another python script:
```python
from infimnist_parser import convert
data = convert(test10k-labels, test10k-patterns, save = False)
```
