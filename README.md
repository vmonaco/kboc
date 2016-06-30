[paper]: http://arxiv.org/pdf/.pdf
## Robust Keystroke Biometric Anomaly Detection

Source code for the submissions made by U.S. Army Research Laboratory to the [Keystroke Biometrics Ongoing Competition](https://sites.google.com/site/btas16kboc/home) (KBOC).

This repository also contains the code to reproduce the results in the companion [paper][], where the anomaly detection systems are described in detail.

## Instructions

### Dependencies

Results were obtained with the following software versions:

```
> %watermark -v -p numpy,pandas,scikit-learn,tensorflow,pohmm
CPython 3.5.1
IPython 4.1.2

numpy 1.11.0
pandas 0.18.1
scikit-learn 0.17.1
tensorflow 0.8.0
pohmm 0.2
```

It is recommended to use [Anaconda](https://www.continuum.io/downloads) and create a virtual env with the above dependencies installed.

To reproduce the main results, place the KBOC databases (zip files) in the data/ folder. Then run the main.py script:

```
> python main.py
```

This will create the validation and submission files for 21 different systems. Systems 1-15 were submitted to the KBOC. Note that this script may take several hours to complete, depending on the CPU, use of GPU for neural network training, and available memory. The resulting validation and submission score files may also vary slightly from those in the repository depending on the GPU used for training neural network models.

To plot the score distributions of any system (requires matplotlib and seaborn), use the plot_scores.py script:

```
> python plot_scores.py system6
```

## System descriptions

Default is to use SD score normalization, SD feature normalization, and keystroke correspondence between the given and target sequence (described in the [paper][paper]).

#### System 1
Deep autoencoder with three hidden layers of dimensions 5, 4, and 3.

#### System 2
Variational autoencoder with two hidden layers of dimension 5.

#### System 3
Partially observable hidden Markov model with 2 hidden states and lognormal emissions.

#### System 4
One-class support vector machine (SVM) using press-press latency and duration features.

#### System 5
Contractive autoencoder with hidden layer of dimension 400.

#### System 6
Manhattan distance.

#### System 7
Autoencoder with a single hidden layer of dimension 5.

#### System 8
Contractive autoencoder with hidden layer of dimension 200.

#### System 9
Mean ensemble of systems 3, 4, and 5.

#### System  10
Mean ensemble of systems 1-8.

#### System 11
Same as system 3, except using min/max score normalization.

#### System 12
Same as system 4, except using min/max score normalization.

#### System 13
Same as system 5, except using min/max score normalization.

#### System 14
Same as system 8, except using min/max score normalization.

#### System 15
Mean ensemble of systems 11-14.

### Not submitted 

#### System 16
Manhattan distance using min/max score normalization.

#### System 17
Manhattan distance using no score normalization.

#### System 18
Manhattan distance without the keystroke alignment.

#### System 19
Manhattan distance without keystroke alignment and using min/max score normalization.

#### System 20
Manhattan distance without keystroke alignment and using no score normalization.

#### System 21
Manhattan distance using min/max feature normalization.

#### System 22
Manhattan distance discarding modifier keys.

#### System 23
Manhattan distance discarding modifier keys and using min/max score normalization.

#### System 24
Manhattan distance discarding modifier keys and using no score normalization.