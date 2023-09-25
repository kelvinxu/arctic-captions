# arctic-captions

Source code for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)
runnable on GPU and CPU.

Joint collaboration between the Université de Montréal & University of Toronto.

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* A relatively recent version of [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [argparse](https://www.google.ca/search?q=argparse&oq=argparse&aqs=chrome..69i57.1260j0j1&sourceid=chrome&es_sm=122&ie=UTF-8#q=argparse+pip)

In addition, this code is built using the powerful
[Theano](http://www.deeplearning.net/software/theano/) library. If you
encounter problems specific to Theano, please use a commit from around
February 2015 and notify the authors.

To use the evaluation script (metrics.py): see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## How to Install Dependencies
Go to your terminal or command line and type each of the commands below:

```
pip install numpy
pip install scikit-learn
pip install scikit-image
pip install argparse
pip install theano
```
## Common Issues and How to Solve them
### cPickle import issue
In some files there will be an import statement like this:

`import cPickle as pkl`

that might give you an error statement when you try to run any files within this repository

If you have tried: 

`pip install cPickle`

and that command results in an error message:

```
ERROR: Could not find a version that satisfies the requirement cPickle (from versions: none)
ERROR: No matching distribution found for cPickle
```
Then change "cPickle" to "pickle" as shown below:

`import pickle as pkl`

cPickle is just a faster version of pickle, so this should work as an alternative

### Theano Issue
As the `Dependencies` section stated, if you ran into issues with Theano then you would need to revert back to a previous commit.

To do this, first do:

`git log`

This will show the previous commits, if you want to see more commits then press the spacebar or the enter key. Once you found the right commit,
press "q" to get out of `git log`.

Then, copy the combination of letters and digits after "commit" on the commit you want to revert back to, for example for `commit de281cebfb9965cc53873f9dd92e6eca418dc4a7` you only want to copy `de281cebfb9965cc53873f9dd92e6eca418dc4a7`.

Then type in `git reset` and paste the thing you copied a space after that, for example:

`git reset de281cebfb9965cc53873f9dd92e6eca418dc4a7`

And that should revert you back to a previous copy.

### Errors with print statements
If you are having errors related to some print statements in this codebase, then you should put the strings/variables in the same line as `print` inside of parentheses

## Reference

If you use this code as part of any published research, please acknowledge the
following paper (it encourages researchers who publish their code!):

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *To appear ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    } 

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).
