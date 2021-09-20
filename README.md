# EPR
Implementation of Weakly Supervised Explainable Phrasal Reasoningfor Natural Language Inference

## Execution
```console
python data_process.py --do_train --do_dev --do_test
python aligner.py --do_train --do_dev --do_test
python main_train.py
python explain_phrase.py
```

## Evaluation
```console
python micro_eva.py
```
