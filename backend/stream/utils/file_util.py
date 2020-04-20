import os


def get_dir_paths(dir_path):
    file_path_list = list()
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isdir(file_path) and not file_name.startswith('.'):
            file_path_list.append(file_path)
    return file_path_list


def get_file_paths(dir_path, ext=None):
    file_path_list = list()
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and (ext is None or file_name.endswith(ext)):
            file_path_list.append(file_path)
    return file_path_list


def make_dirs(dir_path):
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_parent_dirs(file_path):
    dir_path = os.path.dirname(file_path)
    make_dirs(dir_path)
