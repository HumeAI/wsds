import fire

from .ws_tools import commands

def main():
    fire.Fire(commands, name="wsds")

if __name__ == '__main__':
    main()
