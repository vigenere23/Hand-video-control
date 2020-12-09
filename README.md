# Project GIF-7001 - Hand recognition

## Setup

### Dataset

1. Download CSV datasets from https://www.kaggle.com/datamunge/sign-language-mnist.
2. Extract and put the CSVs inside `/data`.

> These CSVs will only be used in the first run. They will then be converted to binaries for faster reuse.

### Python

It is highly recommanded to create a standard venv using the following command:

```bash
python -m venv .venv
```

> :warning: On Linux, the command `python` might be replaced with `python3` to use the latest version

Then, to use it, simply run:

```bash
source .venv/bin/activate
```

inside the terminal on which you will run the scripts.

> :warning: If the previous command needed `python3`, after setuping the venv, simply writting `python` will work.

## Run

To run a script, you first need to setup the venv as described before. Then, any script can be run with:

```bash
python <script_name>.py
```

or in module mode with:

```
python -m <script_name>
```

> Notice that the extension is no longer present in module mode

> :warning: This has not yet been tested

There are different scripts that can be run:

1. `training`: this will train the convotutional neural network and save a snapshot of its best state.
