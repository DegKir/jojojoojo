from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

# Загрузка сетки и меток границ
mesh = Mesh("mesh/ventricle.xml")
bdry = MeshFunction("size_t", mesh, "mesh/ventricle_facet_region.xml")
ds = Measure("ds", subdomain_data=bdry)
n = FacetNormal(mesh)

# Метки границ
BASE = 1  # Срез основания
ENDO = 2  # Эндокард
EPI = 3   # Эпикард

# Параметры задачи
mu = Constant(10.0)     # кПа
b = Constant(1.0)
T = 20.0                # мс
dt = 1.0                # мс

def p_0(t):
    """Давление на эндокарде как функция времени"""
    return 10.0 * t / T  # кПа

# ВЫБОР СМЕШАННОГО ЭЛЕМЕНТА (тестируем P2-P1 и P1-P1)
# Вариант 1: P2-P1 (рекомендуемая пара для несжимаемости)
P2v = VectorElement("CG", mesh.ufl_cell(), 2)  # P2 для перемещений
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)   # P1 для давления
element_mixed = P2v * P1

# Вариант 2: P1-P1 (может иметь проблемы с устойчивостью)
# P1v = VectorElement("CG", mesh.ufl_cell(), 1)  # P1 для перемещений
# P1 = FiniteElement("CG", mesh.ufl_cell(), 1)    # P1 для давления
# element_mixed = P1v * P1

# Смешанное пространство функций (перемещения + давление)
Wh = FunctionSpace(mesh, element_mixed)
print(f"Размерность смешанного пространства: {Wh.dim()}")

# Функции для решения
wp = Function(Wh)            # (u, p) - вектор перемещений и давление
dwp = TrialFunction(Wh)      # Пробная функция
(wT, pT) = TestFunctions(Wh) # Тестовые функции

# Разделяем на компоненты
(u, p) = split(wp)

# Граничное условие: закрепленное основание (только для перемещений!)
bc_base = DirichletBC(Wh.sub(0), Constant((0.0, 0.0, 0.0)), bdry, BASE)

# Начальные условия (нулевые перемещения, нулевое давление)
wp.assign(interpolate(Constant((0.0, 0.0, 0.0, 0.0)), Wh))

# Кинематика деформации
ndim = u.geometric_dimension()
I = Identity(ndim)
F = I + grad(u)          # Градиент деформации
F = variable(F)          # Для дифференцирования
C = F.T * F              # Тензор Коши-Грина
J = det(F)               # Якобиан (изменение объема)
invF = inv(F)            # Обратный градиент деформации

# Энергетический потенциал для постановки II
E = (C - I) / 2.0        # Тензор деформации Грина-Лагранжа
Q = b * inner(E, E)      # Q = b * tr(E^2)
psi_e = mu/2.0 * (exp(Q) - 1.0)  # Упругий потенциал
psi_p = - p * J          # Потенциал для множителя Лагранжа (p - давление)
psi = psi_e + psi_p      # Полный потенциал

# Первый тензор Пиолы-Кирхгофа (теперь включает давление)
# P = ∂ψ/∂F = ∂ψ_e/∂F - p * J * F^{-T}
P = diff(psi_e, F) - p * J * invF.T

# Выражение для внешнего давления на границе
p_ext = Expression("p0", p0=p_0(0.0), degree=0)

# СЛАБАЯ ФОРМА ДЛЯ ПОСТАНОВКИ II
# 1. Уравнение равновесия (виртуальная работа)
# 2. Условие несжимаемости J-1=0 как ограничение
FF = inner(P, grad(wT)) * dx \
     + dot(J * p_ext * invF.T * n, wT) * ds(ENDO) \
     + pT * (J - 1.0) * dx  # Уравнение несжимаемости

# Эпикард (EPI=3) - свободная поверхность P·N = 0 выполняется автоматически

# Якобиан для метода Ньютона
Tang = derivative(FF, wp, dwp)

# Параметры решателя Ньютона
slv_prms = {'newton_solver': {'linear_solver': 'lu',
                               'absolute_tolerance': 1.0e-8,
                               'relative_tolerance': 1.0e-6,
                               'maximum_iterations': 20}}

# Файлы для результатов
file_u = XDMFFile(mesh.mpi_comm(), "displacement_II.xdmf")
file_p = XDMFFile(mesh.mpi_comm(), "pressure_II.xdmf")
for file in [file_u, file_p]:
    file.parameters['rewrite_function_mesh'] = False
    file.parameters["flush_output"] = True

# Временной цикл
t = 0.0
while t <= T + 1e-10:
    print(f"Время t = {t:.1f} мс")
    
    # Обновляем внешнее давление
    p_ext.p0 = p_0(t)
    
    # Решаем нелинейную задачу
    solve(FF == 0, wp, J=Tang, bcs=[bc_base], solver_parameters=slv_prms)
    
    # Извлекаем и сохраняем решения
    u_sol, p_sol = wp.split()
    
    u_sol.rename("u", "u")
    file_u.write(u_sol, t)
    
    p_sol.rename("p", "p")
    file_p.write(p_sol, t)
    
    # Переход к следующему временному шагу
    t += dt
    if t > T:
        t = T

file_u.close()
file_p.close()
print("Расчет Постановки II завершен")