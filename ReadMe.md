This repository contains the code to reproduce the experiments presented in the paper with the title:

    "Best-Effort Policies for Robust Markov Decision Processes"

## Installation
After downloading this folder onto your machine, please run the following steps to install the required dependencies.
We have tested the code base Python version 3.12.

### 1. Install PRISM

To run the code, the probabilistic model checker PRISM is required. For details, we refer to [the PRISM website](https://www.prismmodelchecker.org/manual/InstallingPRISM/Instructions).

We assume the git is installed on your machine. Then, the following dependencies must be installed on your machine:

1. Java Development Kit (required to run PRISM, see https://prismmodelchecker.org/manual/InstallingPRISM/Instructions for details). On Linux, this can be installed using the commands:

   ```bash
   sudo apt install default-jdk
   ```

   On MacOS, the Java Development kit can be installed via, for example, Homebrew or Oracle (https://www.oracle.com/java/technologies/downloads/#jdk22-mac).

3. PRISM - In the desired PRISM installation folder, clone PRISM (version `4.8.1`) from git and run the makefile:

   ```bash
   git clone https://github.com/prismmodelchecker/prism.git prism --branch v4.8.1;
   cd prism/prism; 
   make
   ```

   For more details on installing and using PRISM, we refer to the PRISM documentation on 
   https://www.prismmodelchecker.org


### 2. Install Python dependencies
Install the remaining dependencies with:

```bash
pip3 install -r requirements.txt
```

## Reproduce experimental results
To reproduce the results presented in the paper, run the `run.py` file as follows:

```
python run.py --prism_location '<path_to_prism_executable>'
```

Replace the prism_location with the appropriate path on your machine. Typically, this path looks something like `/.../prism/bin/prism`, that is, the PRISM installation has a `bin` folder, and the executable is located within that folder.

Expected runtime: 1 hour for the IMDP experiments with PRISM, and 7 hours for the RMDP experiments (tested on an Apple Macbook, M4 Pro Chip, 24 GB of RAM).

### Inspecting results
Upon running the file above, the results presented in the paper are reproduced. These results consist of:

1. A table with the results for the comparison to PRISM on IMDPs
2. A table with the results for robust value iteration for s-rectangular RMDPs

Both tables are exported to the `results/` folder in `.csv` and `.tex` format. The tables presented in the paper are directly obtained from these LaTeX files.

Finally, the PRISM models themselves are exported to the `prism/` folder. These are not reported in the paper, but can be inspected if you are interested in the structure of the IMDP models.