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
