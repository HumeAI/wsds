import fire

from . import ws_tools

def main():
    fire.Fire({k:getattr(ws_tools, k) for k in ws_tools.__all__}, name="wsds")

if __name__ == '__main__':
    main()
