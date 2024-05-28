import json
import argparse
from trainer import train



def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/finetune.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--model_name', type=str, default="finetune")
    parser.add_argument('--convnet_type', type=str, default='resnet32')
    parser.add_argument('--second_task_freeze_stage', type=int, default=0)
    parser.add_argument('--logfilename', type=str, default='experiment_freeze')
    
    return parser


if __name__ == '__main__':
    main()
