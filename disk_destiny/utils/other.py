import inspect
import subprocess
from .banner import banner


def clear_screen() -> None:
    subprocess.run('clear')
    
def print_banner() -> None:
    print(banner)
    
def print_pure_banner() -> None:
    clear_screen()
    print_banner()
    