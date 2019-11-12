## Installation
Run the following inside a python3 virtual environment
```
pip install -r requirements.txt
```

## Test:
Should print out dataset statistics if all is fine.
```
python test.py
```

## Evaluation metric:
Train and Predict functions in trainer_prob.py print out metrics like f1-score and precision.
However, Match_m in Utils/Evalution2.py is used as the main evaluation metric.
