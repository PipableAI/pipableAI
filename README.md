# Pipable 🧪

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