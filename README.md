# IGGCN

Code & data accompanying the paper ["Classification of Epileptic Seizures in EEG Data based on Iterative Gated Graph Convolution Network"](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1454529/full).




## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.


### Data preprocess
* Cd into the `Preprocess` folder
* Start preprocessing the raw data

    ```
         prepare_data_for_tusz.ipynb
    ```

The processed data will be stored in the `data` folder

### Run the IGGCN models

* Cd into the `src` folder
* Run the IDGL model and report the performance

    ```
         python main.py -config config/tusz_eeg/4_class/ggnn.yml
    ```



* Notes: 
    - You can find the output data in the `out` folder specified in the config file.
    - You can download the TUSZ Datasets from [here](https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/).

  
