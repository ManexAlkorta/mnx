# mnx

***mnx*** is a code to handle crystallographic structures and dynamical matrixes. At the moment it is implemented to work with ***QuantumESPRESSO*** and ***VASP*** formats, also supporting the ***SSCHA*** code.

## Installation

It is recommended to prepare a clean python environment to install ***mnx***.

```bash
cd ~/path_to_your_venvs
python -m venv mnx
source mnx/bin/activate
cd ~/path_to_installation_dir
git clone https://github.com/ManexAlkorta/mnx.git
cd mnx
pip install ./ 
deactivate
```

For jupyter lab users, the following lines install the mnx kernel with interactive utilities.

```bash
source ~/path_to_your_venvs/mnx/bin/activate 
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=mnx
jupyter kernelspec list
pip install --upgrade matplotlib ipympl
deactivate
```