from __future__ import print_function

import os
import argparse
from collections import defaultdict
import sys
import re

import numpy as np


def parse_report_file(in_lines, in_eval_domain):
    result = defaultdict(lambda: 0)
    for line in in_lines:
        match = re.match('Domain: {} BLEU (.+) entity precision (.+) recall (.+) and f1 (.+)'.format(in_eval_domain), line)
        if match:
            result['BLEU'] = float(match.group(1))
            result['Ent_P'], result['Ent_R'], result['Ent_F1'] = list(map(float, [match.group(2), match.group(3), match.group(4)]))
            break
    return result


def gather_metrics(in_src_folder, in_eval_domain):
    result_gathered = defaultdict(lambda: [])

    for session_folder in os.listdir(in_src_folder):
        report_files = [name
                        for name in os.listdir(os.path.join(in_src_folder, session_folder))
                        if os.path.splitext(name)[1] == '.txt']
        assert len(report_files) == 1
        with open(os.path.join(in_src_folder, session_folder, report_files[0])) as report_in:
            report_content = report_in.readlines()
        report = parse_report_file(report_content, in_eval_domain)
        for key, value in report.items():
            result_gathered[key].append(value)
    return result_gathered


def write_report(in_report, in_domain, in_output_stream):
    value_lens = [len(value) for value in in_report.values()]
    assert len(set(value_lens)) == 1
    print('Domain: {}, {} measurements'.format(in_domain, value_lens[0]))
    fields = ['BLEU', 'Ent_P', 'Ent_R', 'Ent_F1']
    print('\t'.join(fields))
    field_values = []
    for field in fields:
        value = in_report[field]
        field_values.append('{:.3f}+/-{:.3f}'.format(np.mean(value), np.std(value)))
    print('\t'.join(field_values), file=in_output_stream)


def get_option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('eval_domain')

    return parser


if __name__ == '__main__':
    parser = get_option_parser()
    args = parser.parse_args()
    results = gather_metrics(args.input_folder, args.eval_domain)
    write_report(results, args.eval_domain, sys.stdout)

