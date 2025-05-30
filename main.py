import tkinter as tk
from interface import DisplayFrame, PointsEntry
from math_exp4 import main

A = -1; B = 1; C = -2; D = 2; T = 10 # обмеження
limits = [A, B, C, D, T]

init_conds = init_conds_dt = bound_conds = None
i_conds_expr = i_conds_expr_dt = b_conds_expr = None

def get_limits(entry1, entry2, entry3):
    init_conds, i_conds_expr = entry1.get_cond()
    init_conds_dt, i_conds_expr_dt = entry2.get_cond()
    bound_conds, b_conds_expr = entry3.get_cond()
    print(init_conds, init_conds_dt, bound_conds, sep="\n")
    main(init_conds, i_conds_expr, init_conds_dt, i_conds_expr_dt, bound_conds, b_conds_expr)


# запускаємо інтерфейс
root = tk.Tk()
root.geometry("1000x700")
frame1 = DisplayFrame(root, "Linear differential operator:", "add\\diff_oper.txt", relief="ridge", bd=4)
frame1.pack(fill="x")
frame2 = DisplayFrame(root, "Green's function:               ", "add\\green_func.txt", relief="ridge", bd=4)
frame2.pack(fill="x")

label1 = tk.Label(text=" Initial conditions", font=("Times new roman", 18), anchor='w')
label1.pack(fill="x")
entry1 = PointsEntry(root, "add\\cond0_y.txt", "func0", limits, relief="ridge", bd=4)
entry1.pack(fill="x")
entry2 = PointsEntry(root, "add\\cond0_diff_y.txt", "diff0", limits, relief="ridge", bd=4)
entry2.pack(fill="x")

label1 = tk.Label(text=" Boundary conditions", font=("Times new roman", 18), anchor='w')
label1.pack(fill="x")
entry3 = PointsEntry(root, "add\\condG_y.txt", "funcG", limits, relief="ridge", bd=4)
entry3.pack(fill="x")

start_button = tk.Button(root, text="Start algorithm", font=("Times new roman", 18), bd=4, command=lambda: get_limits(entry1, entry2, entry3))
start_button.place(x=430, y=560)

root.mainloop()




