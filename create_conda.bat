set root=C:\Users\Nathan\miniconda3
call %root%Scripts\activate.bat %root%

call conda create --prefix ./envs python=3.10.11

call conda install -n myenv pygame

call conda install -n myenv pymunk

call conda activate .\envs


cmd \k