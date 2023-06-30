# Pipable 🧪

## Creating a virtual environment (Optional)

If somehow installation fails because of library conflicts, it is suggested to create a virtual environment first then proceed with the installation. 

1) Run `pip install virtualenv`. This will install virtualenv package on your machine.
2) In the home directory, run `python -m env`. This will create a new virtual Environment by the name env.
3) Run `source env/bin/activate` to activate the new virtual environment.

## Installation instructions:

1) Clone the repository
2) Run `pip install -r requirements.txt` within the directory

Any jupyter notebook or python file made within the directory can now import and use Pipable.

3) Create a jupyter notebook or python file within the directory
4) Import Pipable into your file using `from pipable import Pipable`

You can now use the `Pipable` class as follows:

```python
from pipable import Pipable

a = Pipable(pathToCSV="sample_data/alyf.csv")

a.ask("Get all patient ids and vital in the form of table that have vitals as Heart Rate and value between 100 to 150 between march to april 2023")
```

You can now ask pipable anything. Enjoy 🥳