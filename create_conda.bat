set root=C:\Users\Nathan\miniconda3
call %root%Scripts\activate.bat %root%

REM md envs
call conda env create --prefix=envs --file=environment.yml

call conda activate ./envs


cmd \k