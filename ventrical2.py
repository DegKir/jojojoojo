from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

# Загрузка сетки и меток границ
mesh = Mesh("mesh/ventricle.xml")  # Предполагаемое имя файла
bdry = MeshFunction("size_t", mesh, "mesh/ventricle_facet_region.xml")
ds = Measure("ds", subdomain_data=bdry)
n = FacetNormal(mesh)

# Метки границ (согласно заданию)
BASE = 1  # Срез основания
ENDO = 2  # Эндокард
EPI = 3   # Эпикард

# Параметры задачи
mu = Constant(10.0)     # кПа
b = Constant(1.0)
kappa = Constant(1000.0)  # кПа (штрафной параметр)
T = 20.0                # мс
dt = 1.0                # мс

def p_0(t):
    """Давление на эндокарде как функция времени"""
    return 10.0 * t / T  # кПа

# Выбор конечного элемента для перемещений (тестируем P1 и P2)
# Вариант 1: P1 элементы (раскомментировать одну из строк)
element_u = VectorElement("CG", mesh.ufl_cell(), 1)  # P1
# Вариант 2: P2 элементы
# element_u = VectorElement("CG", mesh.ufl_cell(), 2)  # P2

# Пространство функций (только перемещения)
Vh = FunctionSpace(mesh, element_u)
print(f"Размерность пространства: {Vh.dim()}")

# Функции для решения
u = Function(Vh)        # Перемещение (текущее)
du = TrialFunction(Vh)  # Пробная функция
uT = TestFunction(Vh)   # Тестовая функция

# Граничное условие: закрепленное основание
bc_base = DirichletBC(Vh, Constant((0.0, 0.0, 0.0)), bdry, BASE)

# Начальное условие (нулевые перемещения)
u.assign(interpolate(Constant((0.0, 0.0, 0.0)), Vh))

# Кинематика деформации
ndim = u.geometric_dimension()
I = Identity(ndim)
F = I + grad(u)          # Градиент деформации
F = variable(F)          # Для дифференцирования
C = F.T * F              # Тензор Коши-Грина
J = det(F)               # Якобиан (изменение объема)
invF = inv(F)            # Обратный градиент деформации

# Энергетический потенциал для постановки I
E = (C - I) / 2.0        # Тензор деформации Грина-Лагранжа
Q = b * inner(E, E)      # Q = b * tr(E^2)
psi_e = mu/2.0 * (exp(Q) - 1.0)  # Упругий потенциал
psi_p = kappa/4.0 * (J - 1.0)**2  # Штрафной потенциал
psi = psi_e + psi_p      # Полный потенциал

# Первый тензор Пиолы-Кирхгофа
P = diff(psi, F)         # P = ∂ψ/∂F

# Выражение для давления (зависит от времени)
p_exp = Expression("p0", p0=p_0(0.0), degree=0)

# СЛАБАЯ ФОРМА ДЛЯ ПОСТАНОВКИ I
# Уравнение равновесия + граничное условие давления на эндокарде
FF = inner(P, grad(uT)) * dx \
     + dot(J * p_exp * invF.T * n, uT) * ds(ENDO)

# Эпикард (свободная поверхность) не требует явного задания -
# естественное граничное условие P·N = 0 выполняется автоматически

# Якобиан для метода Ньютона
Tang = derivative(FF, u, du)

# Параметры решателя Ньютона
slv_prms = {'newton_solver': {'linear_solver': 'lu',
                               'absolute_tolerance': 1.0e-8,
                               'relative_tolerance': 1.0e-6,
                               'maximum_iterations': 20}}

# Файл для результатов
fileO = XDMFFile(mesh.mpi_comm(), "results_I.xdmf")
fileO.parameters['rewrite_function_mesh'] = False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# Временной цикл
t = 0.0
while t <= T + 1e-10:
    print(f"Время t = {t:.1f} мс")
    
    # Обновляем давление на текущем временном шаге
    p_exp.p0 = p_0(t)
    
    # Решаем нелинейную задачу
    solve(FF == 0, u, J=Tang, bcs=[bc_base], solver_parameters=slv_prms)
    
    # Сохраняем результаты
    u.rename("u", "u")
    fileO.write(u, t)
    
    # Переход к следующему временному шагу
    t += dt
    if t > T:
        t = T

fileO.close()
print("Расчет завершен")