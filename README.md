# Pipable 🧪

## Installation instructions:

1) Clone the repository
2) Run `pip install -r requirements.txt` within the directory

If the requirements installation fails because of library conflicts, you can create a python virtual environment first and then install the requirements within the virtual environment. 

1) Run `pip install virtualenv`. This will install virtualenv package on your machine.
2) In the home directory, run `python -m env`. This will create a new virtual Environment by the name env.
3) Run `source env/bin/activate` to activate the new virtual environment.
4) Run `pip install -r requirements.txt` to install the requirements within the virtual environment.

Any jupyter notebook or python file made within the directory can now import and use Pipable.

1) Create a jupyter notebook or python file within the directory
2) Import Pipable into your file using `from pipable import Pipable`

You can now use the `Pipable` class as follows:

```python
from pipable import Pipable

a = Pipable(
    dataType="csv",
    pathToData="sample_data/medSample.csv",
    pathToADD="sample_data/medSampleADD.json",
    openaiKEY="OPENAI_API_KEY",
    googleCustomKEY="GOOGLE_CUSTOM_SEARCH_API_KEY",
    googleProgrammableKEY="GOOGLE_PROGRAMMABLE_SEARCH_ENGINE_API_KEY"
)

a.ask("Get all patient ids and vital in the form of table that have vitals as Heart Rate and value between 100 to 150 between march to april 2023")

outputs = a.get_all_outputs()
#List of all the outputs will be generated.

```

> `ADD` stands for Action Description Dictionary.

You can now ask pipable anything. Enjoy 🥳
