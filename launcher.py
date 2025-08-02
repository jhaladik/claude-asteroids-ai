"""
AI Game Framework Launcher
Easy interface to run different components
"""

import os
import sys
import subprocess

def print_menu():
    print("\n" + "="*60)
    print("AI GAME FRAMEWORK LAUNCHER")
    print("="*60)
    print("\n1. Desktop Dashboard (pygame interface)")
    print("2. Web Dashboard (browser interface)")
    print("3. Train Agents")
    print("4. Play Asteroids Demo")
    print("5. Play Snake Demo")
    print("6. Transfer Learning Demo")
    print("7. Check Model Status")
    print("8. Quick Train (50 episodes)")
    print("0. Exit")
    print("\n" + "="*60)

def run_command(cmd, cwd="game_ai_framework"):
    """Run a command in the game framework directory"""
    full_path = os.path.join(os.path.dirname(__file__), cwd)
    subprocess.run([sys.executable] + cmd.split(), cwd=full_path)

def main():
    while True:
        print_menu()
        choice = input("\nSelect an option: ")
        
        if choice == "1":
            print("\nLaunching Desktop Dashboard...")
            run_command("dashboard_fixed.py")
        
        elif choice == "2":
            print("\nLaunching Web Dashboard...")
            print("Open http://localhost:5000 in your browser")
            run_command("web_dashboard.py")
        
        elif choice == "3":
            print("\nTraining Options:")
            print("1. Train All Agents")
            print("2. Train Avoidance Agent")
            print("3. Train Combat Agent")
            print("4. Train Navigation Agent")
            print("5. Train Meta Agent")
            
            train_choice = input("\nSelect training option: ")
            
            if train_choice == "1":
                run_command("training/train_agents.py --task all --episodes 500")
            elif train_choice == "2":
                run_command("training/train_agents.py --task avoidance --episodes 500")
            elif train_choice == "3":
                run_command("training/train_agents.py --task combat --episodes 500")
            elif train_choice == "4":
                run_command("training/train_agents.py --task navigation --episodes 500")
            elif train_choice == "5":
                run_command("training/train_agents.py --task meta --episodes 500")
        
        elif choice == "4":
            print("\nLaunching Asteroids Demo...")
            run_command("main.py asteroids")
        
        elif choice == "5":
            print("\nLaunching Snake Demo...")
            run_command("main.py snake")
        
        elif choice == "6":
            print("\nLaunching Transfer Learning Demo...")
            run_command("simple_transfer_demo.py")
        
        elif choice == "7":
            print("\nChecking Model Status...")
            run_command("check_models.py")
            input("\nPress Enter to continue...")
        
        elif choice == "8":
            print("\nRunning Quick Training (50 episodes)...")
            run_command("quick_train.py")
        
        elif choice == "0":
            print("\nExiting...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    print("Welcome to the AI Game Framework!")
    print("This launcher helps you run different components easily.")
    main()