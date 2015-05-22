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
