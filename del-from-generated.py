import os
from collections import defaultdict

folder = "generated"

iters_for_file_names = defaultdict(list)
for d in os.listdir(folder):
    if "iter" in d:
        file_name_without_iter = d[:d.find("iter") - 1] + ".pkl"
        if file_name_without_iter in os.listdir(folder):
            os.remove(folder + "/" + d)
        else:
            iters_for_file_names[file_name_without_iter].append(int(d[d.find("iter") + 5: d.find(".pkl")]))
for file_name in iters_for_file_names:
    max_iter_file = file_name[:file_name.find(".pkl")] + "-iter-" + str(max(iters_for_file_names[file_name])) + ".pkl"
    for it in iters_for_file_names[file_name]:
        iter_file = file_name[:file_name.find(".pkl")] + "-iter-" + str(it) + ".pkl"
        if iter_file != max_iter_file:
            os.remove(folder + "/" + iter_file)