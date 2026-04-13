# WFLW 5-Point Landmark Training

## Setup

```bash
pip install -r requirements.txt
```

## Configure paths

Edit:
```
configs/default.yaml
```

Set:
```
data.data_root
data.train_list
data.val_list
```

## Train
```
python train.py
```

## Evaluate
```
python eval.py
```
## Infer
```
python infer.py --image /path/to/test.jpg --output pred.jpg
```

### Notes

This implementation converts WFLW 98 landmarks into 5 landmarks:
- left eye center
- right eye center
- nose tip
- left mouth corner
- right mouth corner