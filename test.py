import argparse
from record import read_record, calculate_consistency_acc, align_mmvet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.6-7b")
    parser.add_argument("--dataset", type=str, default="mm-vet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lecture", type=bool, default=False)
    parser.add_argument("--reasoning", type=str, default="DMAD")
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    
    args = parser.parse_args()
    logs = read_record(args.dataset, args.model, split = args.split, lecture = args.lecture, reasoning = args.reasoning)
    if args.dataset == "ScienceQA":
        calculate_consistency_acc(args.dataset, args.model, args.reasoning, lecture = args.lecture, split = args.split, num = args.num, agents = args.agents, rounds = args.rounds) 
    elif args.dataset == "mm-vet":
        align_mmvet(args.model, args.reasoning, agents = args.agents, rounds = args.rounds)         
                