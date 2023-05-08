import os

folder = "scored/scores"

for d in os.listdir(folder):
    if "iter" in d:
        file_name_without_iter = d[:d.find("-iter")]
        if file_name_without_iter in os.listdir(folder):
            os.remove(folder + "/" + d)