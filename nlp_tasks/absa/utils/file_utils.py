import os


def write_lines(lines, file_path, mode='w'):
    with open(file_path, mode=mode, encoding='utf-8') as out_file:
        for line in lines:
            out_file.write(line + '\n')


def append_line(line, file_path, mode='a'):
    with open(file_path, mode=mode, encoding='utf-8') as out_file:
        out_file.write(line + '\n')


def read_all_lines(file_path, encoding='utf-8'):
    lines = []
    with open(file_path, encoding=encoding) as in_file:
        for line in in_file:
            lines.append(line.strip())
    return lines


def read_all_lines_generator(file_path, encoding='utf-8'):
    with open(file_path, encoding=encoding) as in_file:
        for line in in_file:
            yield line


def rm_r(file_path):
    """
    remove file recursively
    :param file_path:
    :return:
    """
    for root, dirs, files in os.walk(file_path, topdown=False):
        continue
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
