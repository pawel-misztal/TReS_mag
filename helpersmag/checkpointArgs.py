import argparse
from typing import Tuple

def parseCheckpointArgs() -> Tuple[str,str]:
    args = argparse.ArgumentParser(description="checkpoint parser")
    args.add_argument("--checkpointPath", "-cp" ,dest="checkpointPath", type=str,
                      default=None,
                      help="path to the last checkpoint .pth")
    args.add_argument("--jsonCheckpointPath", "-jcp", dest="jsonCheckpointPath", type=str,
                      default=None,
                      help="path to the .json")
    parsed_args, unknown = args.parse_known_args()
    return parsed_args.checkpointPath, parsed_args.jsonCheckpointPath



if __name__ == '__main__':
    p1, p2 = parseCheckpointArgs()

    print(p1)
    print(p2)
    print(p1==None)
    print(p2==None)

    print("after arggs")