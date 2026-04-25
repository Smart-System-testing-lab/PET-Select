import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='APPS')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=int, default=0.0)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--system_message', type=str, default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--technique', type=str, default='Zeroshot')
    #evaluation
    parser.add_argument('--complexity', type=float, default=0.15879883388495986)
    parser.add_argument('--fold', type = int, default = 1)
    parser.add_argument('--top_k', type = int, default = 1)
    return parser.parse_args()