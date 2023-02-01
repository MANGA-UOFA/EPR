## Execution
### Alignment
```
cd ..
python aligner.py \
--do_test \
--do_train \
--do_dev \
--is_finetune \ 
--_lambda 0.6 \ # global ratio
--encoding_cache_path ./save_encoding/${dataset_name}/token/ \
--save_encoding_path ./save_encoding/${dataset_name}/alignment/
```

### Training
```console
python main_train.py
python explain_phrase.py
```