import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import numpy as np
from sympy import symbols

class DisplayFrame(tk.Frame):
    def __init__(self, master, label_text, file_path, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.file_path = file_path

        self.label = tk.Label(self, text=label_text, font=("Times new roman", 14))
        self.label_latex = self.create_latex_label()

        self.label.pack(side='left',padx=5)
        self.label_latex.pack(side='left', fill='x', padx=5)

    def read_from_file(self):
        with open(self.file_path, 'r') as file:
            text = file.read()
        return text

    def create_latex_label(self):
        label = tk.Label(self, font=("Times new roman", 14))
        fig = matplotlib.figure.Figure((10, 0.7), dpi=100)
        self.subplot = fig.add_subplot(111)
        self.subplot.get_xaxis().set_visible(False)
        self.subplot.get_yaxis().set_visible(False)
        self.subplot.spines['top'].set_visible(False)
        self.subplot.spines['right'].set_visible(False)
        self.subplot.spines['bottom'].set_visible(False)
        self.subplot.spines['left'].set_visible(False)
        self.text_display()

        self.canvas = FigureCanvasTkAgg(fig, label)
        self.canvas.get_tk_widget().pack()

        return label
    def text_display(self, type = None, function = None):
        self.subplot.clear()
        text = self.read_from_file()
        if text != "":
            text = "$"+text+"$"

        self.subplot.text(0.5, 0.5, text, fontsize=16, ha="center")


class PointsEntry(tk.Frame):
    def __init__(self, master, file_path, type, limits, *args, **kwargs): #limits це A, B, C, D, T
        super().__init__(master, *args, **kwargs)
        self.type = type
        self.file_path = file_path
        self.limits = limits
        label_text = ""
        if self.type == "func0":
            label_text = "y(x1, x2, t)|t=0               "
        elif self.type == "diff0":
            label_text = "dy(x1, x2, t)/dt|t=0         "
        elif self.type == "funcG":
            label_text = "y(x1, x2, t)|(x1, x2) ∈ Г"
        self.label = tk.Label(self, text=label_text, font=("Times new roman", 14))

        self.saved_text = self.read_from_file()
        self.text = tk.Text(self, font=("Times new roman", 14), width=40, height=3)
        self.text.insert("1.0", self.saved_text)
        self.text.bind("<KeyRelease>", self.change_text)

        self.button = tk.Button(self, height=3, text="Save", font=("Times new roman", 14), relief="ridge", activebackground="white", bg="white", command=self.save)

        self.label.pack(side='left', padx=5)
        self.text.pack(side='left', fill='x', expand=True)
        self.button.pack(side='left', padx=5)

    def read_from_file(self):
        with open(self.file_path, 'r') as file:
            text = file.read()
        return text
    def save(self):
        with open(self.file_path, 'w') as file:
            file.write(self.text.get("1.0", tk.END))
            self.button.config(bg="white", fg="black")
            self.saved_text = self.text.get("1.0", tk.END)

    def change_text(self, event=None):
        if self.saved_text != self.text.get("1.0", tk.END):
            self.button.config(bg="black", fg="white")
        else:
            self.button.config(bg="white", fg="black")

    def read_lines(self):
        content = self.text.get("1.0", tk.END)
        lines = self.strip(content.strip().split("\n"))
        return lines

    def select_lim(self, els, index, lim_num):
        return float(self.strip(self.replace(els[index].split(","), "[", ""))[lim_num])

    def strip(self, list_of_str):
        for i in range(len(list_of_str)):
            list_of_str[i] = list_of_str[i].strip()
        return list_of_str

    def select_lim(self, els, index):
        val = els[index].strip()
        if val.startswith("["):
            val = val.strip("[]")
            start, end = [float(x.strip()) for x in val.split(",")]
            return start, end
        else:
            const = float(val)
            return const, const

    def get_cond(self):
        t, x1, x2 = symbols('t x1 x2')
        lines = self.read_lines()
        if self.type == "func0" or self.type == "diff0":
            limits = np.zeros(shape=(len(lines), 4))
        elif self.type == "funcG":
            limits = np.zeros(shape=(len(lines), 6))
        Y = []

        A, B, C, D, T = self.limits  # діапазони: x1 ∈ [A, B], x2 ∈ [C, D], t ∈ [0, T]

        for i, line in enumerate(lines):
            els = line.strip("() \n").split(";")
            els = [e.strip() for e in els]

            x1_start, x1_end = self.select_lim(els, 1)
            x2_start, x2_end = self.select_lim(els, 3)

            # Перевірка на x1 та x2
            if not (A <= x1_start <= x1_end <= B):
                raise ValueError(f"x1 limits out of bounds in line {i + 1}: [{x1_start}, {x1_end}] not in [{A}, {B}]")
            if not (C <= x2_start <= x2_end <= D):
                raise ValueError(f"x2 limits out of bounds in line {i + 1}: [{x2_start}, {x2_end}] not in [{C}, {D}]")

            if self.type == "func0" or self.type == "diff0":
                limits[i] = [x1_start, x1_end, x2_start, x2_end]
            elif self.type == "funcG":
                t_start, t_end = self.select_lim(els, 5)

                # Перевірка на t
                if not (0 <= t_start <= t_end <= T):
                    raise ValueError(f"t limits out of bounds in line {i + 1}: [{t_start}, {t_end}] not in [0, {T}]")

                limits[i] = [x1_start, x1_end, x2_start, x2_end, t_start, t_end]

            Y_expr = eval(els[-1])
            Y.append(Y_expr)

        return limits, Y

