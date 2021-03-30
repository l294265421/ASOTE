import os


def write_lines(lines, file_path, mode='w'):
    """

    :param lines:
    :param file_path:
    :param mode:
    :return:
    """
    with open(file_path, mode=mode, encoding='utf-8') as out_file:
        for line in lines:
            out_file.write(line + '\n')


def append_line(line, file_path, mode='a'):
    with open(file_path, mode=mode, encoding='utf-8') as out_file:
        out_file.write(line + '\n')


def read_all_lines(file_path, encoding='utf-8', strip_type='all'):
    lines = []
    with open(file_path, encoding=encoding) as in_file:
        for line in in_file:
            if strip_type == 'all':
                lines.append(line.strip())
            elif strip_type == 'line_separator':
                lines.append(line.strip('\r\n'))
            else:
                lines.append(line)
    return lines


def read_all_lines_generator(file_path, encoding='utf-8'):
    with open(file_path, encoding=encoding) as in_file:
        for line in in_file:
            yield line


def read_all_content(filepath, encoding='utf-8', keep_line_separator=False):
    """

    :param filepath:
    :param encoding:
    :return:
    """
    new_line = None
    if keep_line_separator:
        new_line = ''
    with open(filepath, encoding=encoding, newline=new_line) as in_file:
        return in_file.read()


def rm_r(file_path):
    """
    remove file recursively
    :param file_path:
    :return:
    """
    for root, dirs, files in os.walk(file_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
