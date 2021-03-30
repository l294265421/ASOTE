import datetime


def now(date_format_str='%Y%m%d%H%M%S'):
    return datetime.datetime.now().strftime(date_format_str)


if __name__ == '__main__':
    print(now())
    # 20181110225539
