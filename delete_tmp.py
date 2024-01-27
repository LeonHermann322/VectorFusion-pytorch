import os,shutil

tmp_folder = "/tmp/workdir/vectorfusion/"

for folder in os.listdir(tmp_folder):
    shutil.rmtree(tmp_folder+folder)