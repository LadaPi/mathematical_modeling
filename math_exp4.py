import numpy as np
import sympy as sp
from sympy import pprint
from sympy import lambdify
import scipy as sc
from scipy.integrate import quad, dblquad, tplquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import pi


x1, x2, t, x1_, x2_, t_ = sp.symbols('x1 x2 t, x1_ x2_ t_')
# A, B, C, D = 2, 4, 2, 4
# T = 2
# A_, B_, C_, D_ = 0.5, 5, 1, 5
# T_negative = -1
A, B, C, D = -1, 1, -2, 2
T = 10
A_, B_, C_, D_ = -2, 2, -3, 3
T_negative = -1
c = 100
N1 = 6
N2 = 6
N3 = 6


init_conds = np.matrix([[A, B, C, D]])
init_conds_dt = np.matrix([[A, B, C, D]])
bound_conds = np.matrix([[-1, -1, -2, 2, 0, 10], [-1, 1, -2, -2, 0, 10]])
# bound_conds = np.matrix([[-1, -1, -2, 2, 0, 10], [-1, 1, -2, -2, 0, 10], [1, 1, -2, 2, 0, 10], [-1, 1, 2, 2, 0, 10]])
i_conds_expr = sp.Matrix([- (x1 ** 4 / 12 + x2 ** 4 / 12 + x1 ** 3 * x2 ** 3 / 18)])
i_conds_expr_dt = sp.Matrix([0])
b_conds_expr = sp.Matrix(
    [[t ** 3 / 6 - (1/12 + x2 ** 4 / 12 - x2 ** 3 / 18)], [t ** 3 / 6 - (x1 ** 4 / 12 + 4/3 - 4 / 9 * x1 ** 3)]])
# b_conds_expr = sp.Matrix(
#      [[t ** 3 / 6 - (1/12 + x2 ** 4 / 12 - x2 ** 3 / 18)], [t ** 3 / 6 - (x1 ** 4 / 12 + 4/3 - 4 / 9 * x1 ** 3)],
#      [t ** 3 / 6 - (1/12 + x2 ** 4 / 12 + x2 ** 3 / 18)], [t ** 3 / 6 - (x1 ** 4 / 12 + 4/3 + 4 / 9 * x1 ** 3)]])


class Point:
    def __init__(self, x1, x2, t):
        self.x1 = x1
        self.x2 = x2
        self.t = t

    def print(self, *args, **kwargs):
        print(f"({self.x1:.3f}, {self.x2:.3f}, {self.t:.3f})", *args, **kwargs)


def mult_pcw(expr):  # Внесення множника в Piecewise
    multiplier = expr.args[0]
    if type(multiplier) == sp.Float:
        val = expr.args[1].args[0][0] * multiplier
        cond = expr.args[1].args[0][1]
        return sp.Piecewise((val, cond), (0, True))
    else:
        return expr


def pcw_sum(expr1, expr2):  # Функції з однаковими умовами
    expr1 = sp.sympify(expr1)
    expr2 = sp.sympify(expr2)
    if expr1.has(sp.Piecewise) and expr2.has(sp.Piecewise):
        multiplier1 = expr1.args[0]
        multiplier2 = expr2.args[0]
        if type(multiplier1) == sp.Float:
            val1 = expr1.args[1].args[0][0] * multiplier1
            cond1 = expr1.args[1].args[0][1]
        else:
            val1 = expr1.args[0][0]
            cond1 = expr1.args[0][1]
        if type(multiplier2) == sp.Float:
            val2 = expr2.args[1].args[0][0] * multiplier2
            cond2 = expr2.args[1].args[0][1]
        else:
            val2 = expr2.args[0][0]
            cond2 = expr2.args[0][1]
        return sp.Piecewise(
            (sp.simplify(val1 + val2), cond1),
            (0, True)
        )
    elif expr1.has(sp.Piecewise):
        multiplier1 = expr1.args[0]
        if type(multiplier1) == sp.Float:
            val1 = expr1.args[1].args[0][0] * multiplier1
            cond1 = expr1.args[1].args[0][1]
        else:
            val1 = expr1.args[0][0]
            cond1 = expr1.args[0][1]
        return sp.Piecewise(
            (sp.simplify(val1 + expr2), cond1),
            (expr2, True)
        )
    elif expr2.has(sp.Piecewise):
        multiplier2 = expr2.args[0]
        if type(multiplier2) == sp.Float:
            val2 = expr2.args[1].args[0][0] * multiplier2
            cond2 = expr2.args[1].args[0][1]
        else:
            val2 = expr2.args[0][0]
            cond2 = expr2.args[0][1]
        return sp.Piecewise(
            (sp.simplify(expr1 + val2), cond2),
            (expr1, True)
        )
    else:
        return expr1 + expr2


def Green_func(x1, x2, t):
    inside = c ** 2 * t ** 2 - (x1 ** 2 + x2 ** 2)
    expr = 1 / (2 * sp.pi * c * sp.sqrt(inside))
    cond = inside > 0
    return sp.Piecewise((expr, cond), (0, True))


def Green_diff(x1, x2, t):
    inside = c ** 2 * t ** 2 - (x1 ** 2 + x2 ** 2)
    expr = -c * t / (2 * sp.pi * (sp.sqrt(inside) ** 3))
    cond = inside > 0
    return sp.Piecewise((expr, cond), (0, True))


def u(x1, x2, t):
    #t ** 3 / 6 - (x1 ** 4 / 12 + x2 ** 4 / 12 + x1 ** 3 * x2 ** 3 / 18)
    return t + c ** 2 * (x1**2 + x1 * x2**3 / 3 + x1**3 * x2 / 3 + x2**2)


def mid_rec_y_inf(A, B, C, D, T, N, write=False):
    cond = c ** 2 * t ** 2 - (x1 ** 2 + x2 ** 2) > 0
    S = sp.Piecewise((0, cond), (0, True))
    h1 = (B - A) / N
    h2 = (D - C) / N
    h3 = T / N
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x1_ = A + (i + 0.5) * h1
                x2_ = C + (j + 0.5) * h2
                t_ = (k + 0.5) * h3
                S += Green_func(x1 - x1_, x2 - x2_, t - t_) * u(x1_, x2_, t_)
    res = S * h1 * h2 * h3

    if write:
        with open('y_inf.txt', 'w') as f:
            f.write(str(res))
    return res


def y_inf_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return sp.sympify(content)


def monte_carlo(func, A, B, N):
    S = 0
    for i in range(N):
        x1_ = np.random.uniform(A, B)
        S += func(x1_)
    return S * (B - A) / N


def monte_carlo_dbl(func, A, B, C, D, N):
    S = 0
    for i in range(N):
        x1_ = np.random.uniform(A, B)
        x2_ = np.random.uniform(C, D)
        S += func(x1_, x2_)
    return S * (B - A) * (D - C) / N


def coords_to_points(points_x1, points_x2, points_t):
    x1_n = len(points_x1)
    x2_n = len(points_x2)
    t_n = len(points_t)
    points = np.zeros((x1_n * x2_n * t_n), dtype=Point)
    for i in range(x1_n):
        for j in range(x2_n):
            for k in range(t_n):
                points[i * x2_n * t_n + j * t_n + k] = Point(points_x1[i], points_x2[j], points_t[k])
    return points


def discretize_S0(x1_n, x2_n, t_n):
    points_x1 = np.linspace(A, B, x1_n)
    points_x2 = np.linspace(C, D, x2_n)
    points_t = np.linspace(T_negative, 0, t_n)
    points = coords_to_points(points_x1, points_x2, points_t)
    return points


def DCR_space(a, b, c, d, N):  # spacing of doubly connected regions of type [a, b) U (c, d]
    result = np.zeros(N)
    h = ((b - a) + (d - c)) / N
    x = a
    i = 0
    while x < b:
        result[i] = x
        x += h
        i += 1
    x = d
    while x > c and i < N:
        result[i] = x
        x -= h
        i += 1
    return result


def discretize_SG(A_, A, B, B_, x1_n, C_, C, D, D_, x2_n, T, t_n):
    # x1 in [A_, A) U (B, B_], x2 in [C_, C) U (D, D_], t in (0 T]
    points_x1 = DCR_space(A_, A, B, B_, x1_n)
    points_x2 = DCR_space(C_, C, D, D_, x2_n)
    points_t = np.linspace(1e-5, T, t_n)

    points = coords_to_points(points_x1, points_x2, points_t)
    return points


def build_B(init_conds, init_conds_dt, bound_conds, points_S0, points_SG):
    R_0 = len(init_conds)
    R_0_dt = len(init_conds_dt)
    R_G = len(bound_conds)
    M_0 = len(points_S0)
    M_G = len(points_SG)
    # B - матриця розмірності (M_0 + M_G) x (R_0 + R_G)
    # Матрицю B шукаємо у вигляді блочної матриці із 4 блоків B11, B12, B21, B22 (2 x 2).
    # B11 - матриця розмірності R_0 x M_0, B12 - R_0 x M_G, B21 - R_G x M_0, B22 - R_G x M_G.

    B11 = np.zeros((R_0 + R_0_dt, M_0), dtype=object)
    B12 = np.zeros((R_0 + R_0_dt, M_G), dtype=object)
    B21 = np.zeros((R_G, M_0), dtype=object)
    B22 = np.zeros((R_G, M_G), dtype=object)

    for r in range(R_0):
        for m in range(M_0):
            B11[r, m] = Green_func(x1 - points_S0[m].x1, x2 - points_S0[m].x2, - points_S0[m].t)
        for m in range(M_G):
            B12[r, m] = Green_func(x1 - points_SG[m].x1, x2 - points_SG[m].x2, - points_SG[m].t)

    for r in range(R_0_dt):
        for m in range(M_0):
            B11[r + R_0, m] = Green_diff(x1 - points_S0[m].x1, x2 - points_S0[m].x2, - points_S0[m].t)
        for m in range(M_G):
            B12[r + R_0, m] = Green_diff(x1 - points_SG[m].x1, x2 - points_SG[m].x2, - points_SG[m].t)

    for r in range(R_G):
        for m in range(M_0):
            B21[r, m] = Green_func(x1 - points_S0[m].x1, x2 - points_S0[m].x2, t - points_S0[m].t)
        for m in range(M_G):
            B22[r, m] = Green_func(x1 - points_SG[m].x1, x2 - points_SG[m].x2, t - points_SG[m].t)

    return [[B11, B12], [B21, B22]]


def build_U(points_S0, points_SG):
    M_0 = len(points_S0)
    M_G = len(points_SG)
    U = np.zeros((M_0 + M_G, 1), dtype=object)
    for m in range(M_0):
        U[m, 0] = sp.symbols(f'u0{m + 1}')
    for m in range(M_G):
        U[m + M_0, 0] = sp.symbols(f'uG{m + 1}')
    return U


def to_pcw_sum(expr, expr_pcw):
    print("expr:", expr)
    print("expr_pcw:", expr_pcw)
    print("expr_pcw.args[0]:", expr_pcw.args[1].args[0][0])
    res_val = expr + expr_pcw.args[0][0]
    return sp.Piecewise((res_val, expr_pcw.args[1]), (0, True))


def build_Y(i_conds_expr, i_conds_expr_dt, b_conds_expr):
    R0 = len(i_conds_expr)
    R0_dt = len(i_conds_expr_dt)
    RG = len(b_conds_expr)
    Y = np.zeros((R0 + R0_dt + RG, 1), dtype=object)
    for r in range(R0):
        Y[r, 0] = i_conds_expr[r]
    for r in range(R0_dt):
        Y[r + R0, 0] = i_conds_expr_dt[r]
    for r in range(RG):
        Y[r + R0 + R0_dt, 0] = b_conds_expr[r]
    return Y


def multiply_piecewise(expr1, expr2):
    val1, cond1 = expr1.args[0]
    val2, cond2 = expr2.args[0]
    return sp.Piecewise(
        (sp.simplify(val1 * val2), sp.And(cond1, cond2)),
        (0, True)
    )


def Pij_term1(all_init_conds, B, i, j):
    print("Pij_term1 is calculating...")
    R_0 = len(all_init_conds)
    M_0 = B[0][i].shape[1]
    M_G = B[0][j].shape[1]

    Int_term1 = np.zeros((M_0, M_G), dtype=object)
    B1i = B[0][i]
    B1j = B[0][j]

    for m in range(R_0): # Кількість доданків (матриць; кожна матриця належить своїй області)
        for i in range(M_0):
            for j in range(M_G):
                x1_lower = all_init_conds[m, 0]
                x1_upper = all_init_conds[m, 1]
                x2_lower = all_init_conds[m, 2]
                x2_upper = all_init_conds[m, 3]

                result = 0
                if x1_lower == x1_upper:
                    h = (x2_upper - x2_lower) / N1
                    x2_ = np.linspace(x2_lower, x2_upper, N1)

                    B1i_ = B1i[m, i].subs(x1, x1_lower)
                    B1j_ = B1j[m, j].subs(x1, x1_lower)
                    for k in range(N1):
                        result += B1i_.subs({x2: x2_[k] + h / 2}) * B1j_.subs({x2: x2_[k] + h / 2})
                    result *= h

                elif x2_lower == x2_upper:
                    h = (x1_upper - x1_lower) / N1
                    x1_ = np.linspace(x1_lower, x1_upper, N1)
                    B1i_ = B1i[m, i].subs(x2, x2_lower)
                    B1j_ = B1j[m, j].subs(x2, x2_lower)
                    for k in range(N1):
                        result += B1i_.subs({x1: x1_[k] + h / 2}) * B1j_.subs({x1: x1_[k] + h / 2})
                    result *= h

                else:
                    h1 = (x1_upper - x1_lower) / N2
                    h2 = (x2_upper - x2_lower) / N2
                    x1_ = np.linspace(x1_lower, x1_upper, N2)
                    x2_ = np.linspace(x2_lower, x2_upper, N2)
                    B1i_ = B1i[m, i]
                    B1j_ = B1j[m, j]
                    for k in range(N2):
                        for l in range(N2):
                            result += B1i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2}) * B1j_.subs(
                                {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2})
                    result *= h1 * h2

                Int_term1[i, j] += result
    return Int_term1


def Pij_term2(bound_conds, B, i, j):
    print("Pij_term2 is calculating...")
    R_G = len(bound_conds)
    M_0 = B[1][i].shape[1]
    M_G = B[1][j].shape[1]

    Int_term2 = np.zeros((M_0, M_G), dtype=object)
    B2i = B[1][i]
    B2j = B[1][j]
    for m in range(R_G):
        for i in range(M_0):
            for j in range(M_G):
                x1_lower = bound_conds[m, 0]
                x1_upper = bound_conds[m, 1]
                x2_lower = bound_conds[m, 2]
                x2_upper = bound_conds[m, 3]
                t_lower = bound_conds[m, 4]
                t_upper = bound_conds[m, 5]

                result = 0
                if x1_lower == x1_upper:
                    h1 = (x2_upper - x2_lower) / N2
                    h2 = (t_upper - t_lower) / N2
                    x2_ = np.linspace(x2_lower, x2_upper, N2)
                    t_ = np.linspace(t_lower, t_upper, N2)
                    B2i_ = B2i[m, i].subs(x1, x1_lower)
                    B2j_ = B2j[m, j].subs(x1, x1_lower)
                    for k in range(N2):
                        for l in range(N2):
                            result += B2i_.subs({x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2}) * B2j_.subs(
                                {x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2})
                    result *= h1 * h2

                elif x2_lower == x2_upper:
                    h1 = (x1_upper - x1_lower) / N2
                    h2 = (t_upper - t_lower) / N2
                    x1_ = np.linspace(x1_lower, x1_upper, N2)
                    t_ = np.linspace(t_lower, t_upper, N2)
                    B2i_ = B2i[m, i].subs(x2, x2_lower)
                    B2j_ = B2j[m, j].subs(x2, x2_lower)
                    for k in range(N2):
                        for l in range(N2):
                            result += B2i_.subs({x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2}) * B2j_.subs(
                                {x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2})
                    result *= h1 * h2
                else:
                    h1 = (x1_upper - x1_lower) / N3
                    h2 = (x2_upper - x2_lower) / N3
                    h3 = (t_upper - t_lower) / N3
                    x1_ = np.linspace(x1_lower, x1_upper, N3)
                    x2_ = np.linspace(x2_lower, x2_upper, N3)
                    t_ = np.linspace(t_lower, t_upper, N3)

                    B2i_ = B2i[m, i]
                    B2j_ = B2j[m, j]
                    for k in range(N3):
                        for l in range(N3):
                            for s in range(N3):
                                result += B2i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2}) * B2j_.subs(
                                    {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2})
                    result *= h1 * h2 * h3
                Int_term2[i, j] += result
    return Int_term2


def build_P(all_init_conds, bound_conds, B, write = False):
    print("P is calculating...")
    P11 = Pij_term1(all_init_conds, B, 0, 0) + Pij_term2(bound_conds, B, 0, 0)
    P12 = Pij_term1(all_init_conds, B, 0, 1) + Pij_term2(bound_conds, B, 0, 1)
    P21 = Pij_term1(all_init_conds, B, 1, 0) + Pij_term2(bound_conds, B, 1, 0)
    P22 = Pij_term1(all_init_conds, B, 1, 1) + Pij_term2(bound_conds, B, 1, 1)
    P = np.block([[P11, P12], [P21, P22]])

    if write:
        with open('P.txt', 'w') as f:
            f.write(str(P.tolist()).replace('], [', '],\n['))
    return P


def Byi_term1(all_init_conds, B, Y0, yInf0, j):  # Y0 - R_0 x 1, yInf0 - в точці t = 0
    print("Byi_term1 is calculating...")
    R_0 = len(all_init_conds)
    M_0 = B[0][j].shape[1]

    Int_term1 = np.zeros((M_0, 1), dtype=object)
    B1i = B[0][j]

    for m in range(R_0):
        for i in range(M_0):
            x2_lower = all_init_conds[m, 2]
            x2_upper = all_init_conds[m, 3]
            x1_lower = all_init_conds[m, 0]
            x1_upper = all_init_conds[m, 1]

            result = 0
            diff = Y0[m] - yInf0
            if x1_lower == x1_upper:
                h1 = (x2_upper - x2_lower) / N1
                x2_ = np.linspace(x2_lower, x2_upper, N1)
                B1i_ = B1i[m, i].subs(x1, x1_lower)
                diff = diff.subs(x1, x1_lower)
                for k in range(N1):
                    result += B1i_.subs({x2: x2_[k] + h1 / 2}) * diff.subs({x2: x2_[k] + h1 / 2})
                result *= h1

            elif x2_lower == x2_upper:
                h1 = (x1_upper - x1_lower) / N1
                x1_ = np.linspace(x1_lower, x1_upper, N1)
                B1i_ = B1i[m, i].subs(x2, x2_lower)
                diff = diff.subs(x2, x2_lower)
                for k in range(N1):
                    result += B1i_.subs({x1: x1_[k] + h1 / 2}) * diff.subs({x1: x1_[k] + h1 / 2})
                result *= h1

            else:
                h1 = (x1_upper - x1_lower) / N2
                h2 = (x2_upper - x2_lower) / N2
                x1_ = np.linspace(x1_lower, x1_upper, N2)
                x2_ = np.linspace(x2_lower, x2_upper, N2)
                B1i_ = B1i[m, i]
                for k in range(N2):
                    for l in range(N2):
                        result += B1i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2}) * diff.subs(
                            {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2})
                result *= h1 * h2
            Int_term1[i, 0] += result

    return Int_term1


def Byi_term2(bound_conds, B, YG, yInf, j):  # Y0 - R_G x 1
    print("Byi_term2 is calculating...")
    R_G = len(bound_conds)
    M_G = B[1][j].shape[1]

    Int_term1 = np.zeros((M_G, 1), dtype=object)
    B1i = B[1][j]

    for m in range(R_G):
        for i in range(M_G):
            x2_lower = bound_conds[m, 2]
            x2_upper = bound_conds[m, 3]
            x1_lower = bound_conds[m, 0]
            x1_upper = bound_conds[m, 1]
            t_lower = bound_conds[m, 4]
            t_upper = bound_conds[m, 5]

            result = 0
            diff = YG[m] - yInf
            if x1_lower == x1_upper:
                h2 = (x2_upper - x2_lower) / N2
                h3 = (t_upper - t_lower) / N2
                x2_ = np.linspace(x2_lower, x2_upper, N2)
                t_ = np.linspace(t_lower, t_upper, N2)
                B1i_ = B1i[m, i].subs(x1, x1_lower)
                diff = diff.subs(x1, x1_lower)
                for k in range(N2):
                    for l in range(N2):
                        result += B1i_.subs({x2: x2_[k] + h2 / 2, t: t_[l] + h3 / 2}) * diff.subs(
                            {x2: x2_[k] + h2 / 2, t: t_[l] + h3 / 2})
                result *= h2 * h3

            elif x2_lower == x2_upper:
                h1 = (x1_upper - x1_lower) / N2
                h3 = (t_upper - t_lower) / N2
                x1_ = np.linspace(x1_lower, x1_upper, N2)
                t_ = np.linspace(t_lower, t_upper, N2)
                B1i_ = B1i[m, i].subs(x2, x2_lower)
                diff = diff.subs(x2, x2_lower)
                for k in range(N2):
                    for l in range(N2):
                        result += B1i_.subs({x1: x1_[k] + h1 / 2, t: t_[l] + h3 / 2}) * diff.subs(
                            {x1: x1_[k] + h1 / 2, t: t_[l] + h3 / 2})
                result *= h1 * h3

            else:
                h1 = (x1_upper - x1_lower) / N3
                h2 = (x2_upper - x2_lower) / N3
                h3 = (t_upper - t_lower) / N3
                x1_ = np.linspace(x1_lower, x1_upper, N3)
                x2_ = np.linspace(x2_lower, x2_upper, N3)
                t_ = np.linspace(t_lower, t_upper, N3)
                B1i_ = B1i[m, i]
                for k in range(N3):
                    for l in range(N3):
                        for s in range(N3):
                            result += B1i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2}) * diff.subs(
                                {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2})
                result *= h1 * h2 * h3

            Int_term1[i, 0] += result
    return Int_term1


def build_By(all_init_conds, bound_conds, B, Y0, yInf0, YG, yInf, write=False):
    print("By is calculating...")
    By1 = Byi_term1(all_init_conds, B, Y0, yInf0, 0) + Byi_term2(bound_conds, B, YG, yInf, 0)
    By2 = Byi_term1(all_init_conds, B, Y0, yInf0, 1) + Byi_term2(bound_conds, B, YG, yInf, 1)
    By = np.block([[By1], [By2]])
    if write:
        with open('By.txt', 'w') as f:
            f.write(str(By.tolist()).replace('], [', '],\n['))
    return By


def build_matrix_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
    M = sp.Matrix(eval(content))
    return M


def solve_system(P_pinv, By):
    res = P_pinv * By  # Візьмемо v = 0
    print("P_pinv:")
    pprint(P_pinv)
    print("")
    return res


def y_0(points_S0, u_0):
    result = 0
    for i in range(len(points_S0)):
        result += Green_func(x1 - points_S0[i].x1, x2 - points_S0[i].x2, t - points_S0[i].t) * u_0[i]
    return result


def y_G(points_SG, u_G):
    result = 0
    for i in range(len(points_SG)):
        result += Green_func(x1 - points_SG[i].x1, x2 - points_SG[i].x2, t - points_SG[i].t) * u_G[i]
    return result


def y(y_inf, y_0, y_G):
    return y_inf + y_0 + y_G


def real_sol(x1, x2, t):
    return t ** 3 / 6 - (x1 ** 4 / 12 + x2 ** 4 / 12 + x1 ** 3 * x2 ** 3 / 18)


# def err_term1(Y0, all_init_conds, B):
#     print("err_term1 is calculating...")
#     R_0 = len(all_init_conds)
#     M_0 = Y0.shape[0]
#     M_0 = B[0].shape[1]
#     M_G = B[0].shape[1]
#
#     Int_term1 = np.zeros((M_0, M_G), dtype=object)
#     B1i = B[0][i]
#     B1j = B[0][j]
#
#     for m in range(R_0): # Кількість доданків (матриць; кожна матриця належить своїй області)
#         for i in range(M_0):
#             for j in range(M_G):
#                 x1_lower = all_init_conds[m, 0]
#                 x1_upper = all_init_conds[m, 1]
#                 x2_lower = all_init_conds[m, 2]
#                 x2_upper = all_init_conds[m, 3]
#
#                 result = 0
#                 if x1_lower == x1_upper:
#                     h = (x2_upper - x2_lower) / N1
#                     x2_ = np.linspace(x2_lower, x2_upper, N1)
#
#                     B1i_ = B1i[m, i].subs(x1, x1_lower)
#                     B1j_ = B1j[m, j].subs(x1, x1_lower)
#                     for k in range(N1):
#                         result += B1i_.subs({x2: x2_[k] + h / 2}) * B1j_.subs({x2: x2_[k] + h / 2})
#                     result *= h
#
#                 elif x2_lower == x2_upper:
#                     h = (x1_upper - x1_lower) / N1
#                     x1_ = np.linspace(x1_lower, x1_upper, N1)
#                     B1i_ = B1i[m, i].subs(x2, x2_lower)
#                     B1j_ = B1j[m, j].subs(x2, x2_lower)
#                     for k in range(N1):
#                         result += B1i_.subs({x1: x1_[k] + h / 2}) * B1j_.subs({x1: x1_[k] + h / 2})
#                     result *= h
#
#                 else:
#                     h1 = (x1_upper - x1_lower) / N2
#                     h2 = (x2_upper - x2_lower) / N2
#                     x1_ = np.linspace(x1_lower, x1_upper, N2)
#                     x2_ = np.linspace(x2_lower, x2_upper, N2)
#                     B1i_ = B1i[m, i]
#                     B1j_ = B1j[m, j]
#                     for k in range(N2):
#                         for l in range(N2):
#                             result += B1i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2}) * B1j_.subs(
#                                 {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2})
#                     result *= h1 * h2
#
#                 Int_term1[i, j] += result
#     return Int_term1


def Pij_term2(bound_conds, B, i, j):
    print("Pij_term2 is calculating...")
    R_G = len(bound_conds)
    M_0 = B[1][i].shape[1]
    M_G = B[1][j].shape[1]

    Int_term2 = np.zeros((M_0, M_G), dtype=object)
    B2i = B[1][i]
    B2j = B[1][j]
    for m in range(R_G):
        for i in range(M_0):
            for j in range(M_G):
                x1_lower = bound_conds[m, 0]
                x1_upper = bound_conds[m, 1]
                x2_lower = bound_conds[m, 2]
                x2_upper = bound_conds[m, 3]
                t_lower = bound_conds[m, 4]
                t_upper = bound_conds[m, 5]

                result = 0
                if x1_lower == x1_upper:
                    h1 = (x2_upper - x2_lower) / N2
                    h2 = (t_upper - t_lower) / N2
                    x2_ = np.linspace(x2_lower, x2_upper, N2)
                    t_ = np.linspace(t_lower, t_upper, N2)
                    B2i_ = B2i[m, i].subs(x1, x1_lower)
                    B2j_ = B2j[m, j].subs(x1, x1_lower)
                    for k in range(N2):
                        for l in range(N2):
                            result += B2i_.subs({x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2}) * B2j_.subs(
                                {x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2})
                    result *= h1 * h2

                elif x2_lower == x2_upper:
                    h1 = (x1_upper - x1_lower) / N2
                    h2 = (t_upper - t_lower) / N2
                    x1_ = np.linspace(x1_lower, x1_upper, N2)
                    t_ = np.linspace(t_lower, t_upper, N2)
                    B2i_ = B2i[m, i].subs(x2, x2_lower)
                    B2j_ = B2j[m, j].subs(x2, x2_lower)
                    for k in range(N2):
                        for l in range(N2):
                            result += B2i_.subs({x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2}) * B2j_.subs(
                                {x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2})
                    result *= h1 * h2
                else:
                    h1 = (x1_upper - x1_lower) / N3
                    h2 = (x2_upper - x2_lower) / N3
                    h3 = (t_upper - t_lower) / N3
                    x1_ = np.linspace(x1_lower, x1_upper, N3)
                    x2_ = np.linspace(x2_lower, x2_upper, N3)
                    t_ = np.linspace(t_lower, t_upper, N3)

                    B2i_ = B2i[m, i]
                    B2j_ = B2j[m, j]
                    for k in range(N3):
                        for l in range(N3):
                            for s in range(N3):
                                result += B2i_.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2}) * B2j_.subs(
                                    {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2})
                    result *= h1 * h2 * h3
                Int_term2[i, j] += result
    return Int_term2


def error(Y0, all_init_conds, YG, bound_conds, By, P_pinv):
    term1 = 0
    for i in range(len(all_init_conds)):
        x1_lower = all_init_conds[i, 0]
        x1_upper = all_init_conds[i, 1]
        x2_lower = all_init_conds[i, 2]
        x2_upper = all_init_conds[i, 3]
        result = 0
        if x1_lower == x1_upper:
            h = (x2_upper - x2_lower) / N1
            x2_ = np.linspace(x2_lower, x2_upper, N1)
            Y0i = Y0[i].subs(x1, x1_lower)
            for k in range(N1):
                result += Y0i.subs({x2: x2_[k] + h / 2}) ** 2
            result *= h
        elif x2_lower == x2_upper:
            h = (x1_upper - x1_lower) / N1
            x1_ = np.linspace(x1_lower, x1_upper, N1)
            Y0i = Y0[i].subs(x2, x2_lower)
            for k in range(N1):
                result += Y0i.subs({x1: x1_[k] + h / 2}) ** 2
            result *= h
        else:
            h1 = (x1_upper - x1_lower) / N2
            h2 = (x2_upper - x2_lower) / N2
            x1_ = np.linspace(x1_lower, x1_upper, N2)
            x2_ = np.linspace(x2_lower, x2_upper, N2)
            Y0i = Y0[i]
            for k in range(N2):
                for l in range(N2):
                    result += Y0i.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2}) ** 2
            result *= h1 * h2
        term1 += result
        print(f"term1 for i={i}: {result}")

    term2 = 0
    for i in range(len(bound_conds)):
        x1_lower = bound_conds[i, 0]
        x1_upper = bound_conds[i, 1]
        x2_lower = bound_conds[i, 2]
        x2_upper = bound_conds[i, 3]
        t_lower = bound_conds[i, 4]
        t_upper = bound_conds[i, 5]
        result = 0
        if x1_lower == x1_upper:
            h1 = (x2_upper - x2_lower) / N2
            h2 = (t_upper - t_lower) / N2
            x2_ = np.linspace(x2_lower, x2_upper, N2)
            t_ = np.linspace(t_lower, t_upper, N2)
            Y0i = YG[i].subs(x1, x1_lower)
            for k in range(N2):
                for l in range(N2):
                    result += Y0i.subs({x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2}) * Y0i.subs(
                        {x2: x2_[k] + h1 / 2, t: t_[l] + h2 / 2})
            result *= h1 * h2

        elif x2_lower == x2_upper:
            h1 = (x1_upper - x1_lower) / N2
            h2 = (t_upper - t_lower) / N2
            x1_ = np.linspace(x1_lower, x1_upper, N2)
            t_ = np.linspace(t_lower, t_upper, N2)
            Y0i = YG[i].subs(x2, x2_lower)
            for k in range(N2):
                for l in range(N2):
                    result += Y0i.subs({x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2}) * Y0i.subs(
                        {x1: x1_[k] + h1 / 2, t: t_[l] + h2 / 2})
            result *= h1 * h2

        else:
            h1 = (x1_upper - x1_lower) / N3
            h2 = (x2_upper - x2_lower) / N3
            h3 = (t_upper - t_lower) / N3
            x1_ = np.linspace(x1_lower, x1_upper, N3)
            x2_ = np.linspace(x2_lower, x2_upper, N3)
            t_ = np.linspace(t_lower, t_upper, N3)
            Y0i = YG[i]
            for k in range(N3):
                for l in range(N3):
                    for s in range(N3):
                        result += Y0i.subs({x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2}) * Y0i.subs(
                            {x1: x1_[k] + h1 / 2, x2: x2_[l] + h2 / 2, t: t_[s] + h3 / 2})
            result *= h1 * h2 * h3
        term2 += result

    term3 = (np.transpose(By) * P_pinv * By)
    return sp.N((term1 + term2 - term3)[0, 0], 10)


def plot(func1, func2, t_val, if_3d, title1, title2):
    x1_vals = np.linspace(A, B, 100)
    x2_vals = np.linspace(C, D, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    if if_3d:
        func_l1 = lambdify((x1, x2, t), func1, 'numpy')
        func_l2 = lambdify((x1, x2, t), func2, 'numpy')
        Z1 = func_l1(X1, X2, t_val)
        Z2 = func_l2(X1, X2, t_val)
    else:
        func_l1 = lambdify((x1, x2), func1, 'numpy')
        func_l2 = lambdify((x1, x2, t), func2, 'numpy')
        Z1 = func_l1(X1, X2)
        Z2 = func_l2(X1, X2, t_val)

    # Створення фігури з двома 3D-підграфіками
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    surf1 = ax1.plot_surface(X1, X2, Z1, cmap='viridis')
    surf2 = ax2.plot_surface(X1, X2, Z2, cmap='plasma')

    ax1.set_title(title1)
    ax2.set_title(title2)

    for ax in [ax1, ax2]:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('z')

    plt.tight_layout()
    plt.show()

def main(init_conds, i_conds_expr, init_conds_dt, i_conds_expr_dt, bound_conds, b_conds_expr):
    yInf = mid_rec_y_inf(A, B, C, D, T, 5, write=True)
    print("yInf:")
    pprint(yInf)
    print("")
    points_S0 = discretize_S0(1, 1, 1)
    points_SG = discretize_SG(A_, A, B, B_, 1, C_, C, D, D_, 1, T, 2)
    all_init_conds = np.vstack((init_conds, init_conds_dt))

    B_matrix = build_B(init_conds, init_conds_dt, bound_conds, points_S0, points_SG)
    print("Matrix B:")
    pprint(B_matrix)
    print("")

    P = build_P(all_init_conds, bound_conds, B_matrix, 1)
    # P = build_matrix_from_file('P.txt')
    print("Matrix P:")
    pprint(P)
    print("")


    Y_matrix = build_Y(i_conds_expr, i_conds_expr_dt, b_conds_expr)
    print("Matrix Y:")
    pprint(Y_matrix)
    print("")

    yInf0 = yInf.subs({t: 0})
    all_init_conds = np.vstack((init_conds, init_conds_dt))


    Y0 = Y_matrix[0:len(all_init_conds), 0]
    YG = Y_matrix[len(all_init_conds):, 0]

    By = build_By(all_init_conds, bound_conds, B_matrix, Y0, yInf0, YG, yInf, write=True)
    # By = build_matrix_from_file('By.txt')
    print("Matrix By:")
    pprint(By)
    print("")

    P_numeric = P.astype(np.float64)
    P_pinv = np.linalg.pinv(P_numeric)

    u_sol = solve_system(P_pinv, By)
    print("u_sol:")
    pprint(u_sol)
    print("")

    u0 = u_sol[0:len(points_S0), 0]
    uG = u_sol[len(points_S0):, 0]
    print("u0:")
    pprint(u0)
    print("")
    print("uG:")
    pprint(uG)
    print("")

    y_0_val = y_0(points_S0, u0)
    y_G_val = y_G(points_SG, uG)

    y_sol = y(yInf, y_0_val, y_G_val)


    def y_sol_func(x1, x2, t):
        return y_sol.subs({x1: x1, x2: x2, t: t})


    print("y_sol:")
    pprint(y_sol)
    print("")

    plot(real_sol(x1, x2, t),y_sol_func(x1, x2, t), 0, True, 'Точний розв\'язок при t = 0', 'Обчислений розв\'язок при t = 0')

    plot(real_sol(x1, x2, t),y_sol_func(x1, x2, t), 5, True, 'Точний розв\'язок при t = 5','Обчислений розв\'язок при t = 5')

    plot(real_sol(x1, x2, t), y_sol_func(x1, x2, t), 10, True, 'Точний розв\'язок при t = 10', 'Обчислений розв\'язок при t = 10')


    err = error(Y0, all_init_conds, YG, bound_conds, By, P_pinv)
    print("Error:")
    pprint(err)
