import datetime


def parse_date(date_str):
    if date_str == 'NA':
        return datetime.datetime(9999, 1, 1, 0, 0)
    else:
        if len(date_str) == 10:
            format_str = '%Y-%m-%d'
        else:
            raise Exception("Format for {} not recognized!".format(date_str))
    return datetime.datetime.strptime(date_str, format_str)
