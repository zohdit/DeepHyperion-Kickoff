# DeepHyperion

Input test generator using illumination search algorithm.

> The following instructions apply for **Python 3.7**.

## Dependencies ##

Installing Python Binding to the Potrace library
``` 
$ sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config
``` 

Install pypotrace:

``` 
$ git clone https://github.com/flupke/pypotrace.git
$ cd pypotrace
$ pip install numpy
$ pip install .
``` 

Installing PyCairo and PyGObject

Instructions provided by https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started.


``` 
$ apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
$ apt-get install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev
``` 

Installing Other Dependencies

This tool has other dependencies such as tensorflow and deap.

To easily install the dependencies with pip:

``` 
$ pip install -r requirements.txt
``` 

Otherwise, you can manually install each required library listed in the requirements.txt file using pip.

## Usage ##
### Input ###

* A trained model in h5 format. The default one is in the folder models;
* A list of seeds used for the input generation. In this implementation, the seeds are indexes of elements of the MNIST dataset. The default list is in the file bootstraps_five;
* properties.py containing the configuration of the tool selected by the user.

### Output ###

When the run is finished, the tool produces the following outputs in the logs folder:

* maps representing inputs distribution;
* json files containing the final reports of the run;
* folders containing the generated inputs (in image format).

### Run the Tool ###

Run the command: python mapelites_mnist.py


