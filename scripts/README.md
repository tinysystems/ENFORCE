# Scripts Execution

## Environment setup
First of all, before running the code, it is required to have a running environemnt on which running the following code and these can be a Google Colab one, which can be unstable if the training is long and not suggestable, or Visual Studio Code environment setup, which requires the steps below, isntead Google Colab needs no setup, but you will have to download missing libraries if there are, which should not be the case with the code in this repository.
### VS Code Setup
1. To procede with VS Code way, we should be sure that python is installed on your system:
```bash
python --version
```
If not install the Python extension in Visual Studio Code extension.

2. Now, the virtual environment can be created, however, you should be sure to be inside your project directory and it is not recommended that it is a github repository, because will create an hidden folder called .venv on which all the libraries will be installed.
The line to create this environment is:
```bash
python -m venv venv
venv\Scripts\activate (Windows)
source .venv/bin/activate (Mac/Linux)
```
You may choose to use Conda environment, too, which should be create like:
```bash
conda create -n myenv python=3.12.2
conda activate myenv
```
If the commands were successfully run you should see at the start of the next commandline insert (.venv) or (conda)

3. Now, there should be the interpreter setup, doing the command combination (Ctrl+Shift+P) on VS Code and inserting Python: Select Interpreter and then pick up the interpreter environment just created, so VS Code knows it has to run/debug code inside that environment

4. Inside the activated environment, shall be installed the dependencies saved in the file "requirements.txt" placed in the main repository folder, with a command:
```bash
pip install -r requirements.txt
```
If in a conda environment:
```bash
conda install --file requirements.txt
```
This will install all packages listed in the file and to check the installation, run the following command:
```bash
pip list
```
5. Now, accessing in this environment with the scripts in this folder and clicking run all it will execute the codes with the selected virtual environments

# Scripts Tasks

The following blocks of codes above performs different tasks, more precisely

### TO BE UPDATED
<table>
    <tr>
        <th>Name</th>
        <th>Role</th>
        <th>Status</th>
    </tr>
    <tr>
        <th>1. Teacher Training.ipynb</th>
        <th>Trains the Convolution Neural Network that will be used as teacher model</th>
        <th>🟢</th>
    </tr>
    <tr>
        <th>2. DirectKD.ipynb</th>
        <th>Performs a direct Knowledge Distillation from the teacher model to the final student fixed model</th>
        <th>🟢</th>
    </tr>
    <tr>
        <th>2.5 Testing Direction.ipynb</th>
        <th>Tests if the direction of the model is learned or if there is a bottleneck in any passage</th>
        <th>🟢</th>
    </tr>
    <tr>
        <th>3. CNNtoFullDense.ipynb</th>
        <th>Conversion of the teacher model into a Full Dense structured model that can lead to an easier conversion to the final student fixed model</th>
        <th>🟢</th>
    </tr>
    <tr>
        <th>4. FullDensetoTinyFC.ipynb</th>
        <th>Performs a conversion from the intermediate model Full Dense organised to a final tiny model</th>
        <th>🟡</th>
    </tr>
</table>

To see the full details on the models generated and run and their role go in the folder "models/"
