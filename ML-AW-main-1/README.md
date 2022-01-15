#### configuration `config.json`，

define training set and test set,

```

data_dir: "../../data"，

data_balance: 0,

train_all: false,

train_selected: ["v25v55", "v35v45", "v45v55", "v55v65"]，

test_selcted: ["v25v35", "v35v45"]

```

data_balancing：
```

0：no data balance（Maximum amount of data）

1：Balance the global number of good and bad

2: Balance the global speed data

3: Under each speed, good and bad data balance

4: Under each speed, after the good and bad data are balanced, the data amount of each speed is further balanced (the amount of data is the smallest)

```


parameter define (select network，MLP，CNN，BiLSTM，restnet18，restnet34, dae)
```

"arch": {

        "type": "BiLSTM",
        
        "args": {}
        
```



training：

`python train.py -c config.json`

testing：

`python test.py -c config.json --resume ./saved/model/***/best.pth`


Tensorboard:

`tensorboard --logdir ./saved/log/`

env:
`
torch>=1.8.0
torchvision >= 0.9.0
numpy >= 1.20.3
tqdm
tensorboard>=2.7.0
`
