import tkinter as tk
from ising_app import IsingApp

def main():
    root = tk.Tk()
    app = IsingApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()