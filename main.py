import os
import yaml
import argparse

from model.Processor import Processor

def main():
    parser = Init_parameters()

    # Update parameters by yaml
    args = parser.parse_args()

    if os.path.exists('./configs/' + args.config_id + '.yaml'):

        with open('./configs/' + args.config_id + '.yaml', 'r') as f:

            yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this config: {}'.format(args.config_id))

    # Update parameters by cmd
    args = parser.parse_args()

    # Show parameters
    print('\n************************************************')
    print('The running config is presented as follows:')
    v = vars(args)
    for i in v.keys():
        print('{}: {}'.format(i, v[i]))
    print('************************************************\n')

    p = Processor(args)
    p.start()


def Init_parameters():
    parser = argparse.ArgumentParser(description='kaggle-bengali')

    # Config
    parser.add_argument('--config_id', '-c', type=str, default='')

    return parser


if __name__ == '__main__':
    main()

