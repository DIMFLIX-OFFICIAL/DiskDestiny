import os
import inquirer
import subprocess
from typing import List

from .banner import banner


def clear_screen() -> None:
	subprocess.run('clear')
	
def print_banner() -> None:
	print(banner)
	
def print_pure_banner() -> None:
	clear_screen()
	print_banner()
	
def get_csv_files(path: str) -> List[str]:
	return [f for f in os.listdir(path) if f.endswith('.csv')]


def ask_list(message: str, choices: list, default: str = None) -> str:
	return  inquirer.prompt([inquirer.List(
		name='answer',
		message=message,
		choices=choices,
		default=default
	)])["answer"]

def ask_checkbox(message: str, choices: list, default: list = None) -> List[str]:
	return inquirer.prompt([inquirer.Checkbox(
		name='answer',
		message=message,
		choices=choices,
		default=default
	)])["answer"]
