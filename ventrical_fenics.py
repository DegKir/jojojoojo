from fenics import *
import time

# Параметры компилятора форм
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

# Чтение сетки и меток границы
print("Загрузка сетки...")
mesh = Mesh("ventrical.xml")
bdry = MeshFunction("size_t", mesh, "ventrical_facet_region.xml")
ds = Measure("ds", subdomain_data=bdry)

# Определение границ (из описания)
BASE = 1  # срез
ENDO = 2  # эндокард
EPI = 3   # эпикард

# Параметры задачи
T = 20.0  # мс
dt_val = 1.0  # мс
mu = Constant(10.0)  # кПа
b = Constant(1.0)
kappa = Constant(1000.0)  # кПа

# Функция давления на границе
def p_0(t):
    return 10.0 * (t / T)  # кПа

# Словарь для хранения результатов
results = {}

# ============================================================================
# ВАРИАНТ I: Штрафная формулировка (только u)
# ============================================================================
print("\n" + "="*60)
print("ВАРИАНТ I: Штрафная формулировка")
print("="*60)

def solve_variant_I(element_order):
    """
    Решение варианта I с заданным порядком элементов
    element_order: 1 для P1, 2 для P2
    """
    print(f"\nВариант I с P{element_order} элементами:")
    
    # Создание функционального пространства
    if element_order == 1:
        V = VectorFunctionSpace(mesh, "CG", 1)
    else:  # element_order == 2
        V = VectorFunctionSpace(mesh, "CG", 2)
    
    # Функции для решения
    u = Function(V)  # текущее перемещение
    u_old = Function(V)  # предыдущее перемещение (для сходимости Ньютона)
    
    # Начальные условия
    u_old.vector()[:] = 0.0
    u.vector()[:] = 0.0
    
    # Граничные условия
    bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), bdry, BASE)
    
    # Пробные и тестовые функции
    v = TestFunction(V)
    du = TrialFunction(V)
    
    # Кинематические величины
    I = Identity(3)
    F = I + grad(u)
    J = det(F)
    
    # Тензор деформаций Грина-Лагранжа
    C = F.T * F
    E = (C - I) / 2.0
    
    # Q = b * tr(E²) = b * inner(E, E)
    Q = b * inner(E, E)
    
    # Энергия деформации
    psi_e = mu/2.0 * (exp(Q) - 1.0)
    psi_p = kappa/4.0 * (J - 1.0)**2
    psi = psi_e + psi_p
    
    # Первый тензор Пиолы-Кирхгоффа
    P = diff(psi, F)
    
    # Слабая форма (статическая задача, без инерции)
    # -∇·P = 0
    F_form = inner(P, grad(v)) * dx
    
    # Граничные условия:
    # 1. На эпикарде: P·N = 0 (естественное условие Неймана, автоматически)
    # 2. На эндокарде: P·N = -p_0 * adj(F)^T · N
    t_current = 0.0
    p_current = p_0(t_current)
    
    # adj(F) = J * F^{-T}
    adjF = J * inv(F).T
    F_form += dot(p_current * adjF.T * FacetNormal(mesh), v) * ds(ENDO)
    
    # Производная для метода Ньютона
    J_form = derivative(F_form, u, du)
    
    # Параметры решателя Ньютона
    solver_params = {'newton_solver': {
        'linear_solver': 'lu',
        'absolute_tolerance': 1e-8,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 50,
        'relaxation_parameter': 1.0
    }}
    
    # Временной цикл
    t = 0.0
    max_time_steps = int(T / dt_val) + 1
    failed_step = None
    last_converged_t = 0.0
    
    for step in range(max_time_steps):
        if t > T + 1e-10:
            break
            
        print(f"  Шаг {step}: t = {t:.1f} мс, давление = {p_current:.2f} кПа")
        
        # Обновление давления на границе
        p_current = p_0(t)
        
        # Переопределение формы с новым давлением
        F_form = inner(P, grad(v)) * dx + dot(p_current * adjF.T * FacetNormal(mesh), v) * ds(ENDO)
        J_form = derivative(F_form, u, du)
        
        try:
            # Решение нелинейной задачи методом Ньютона
            solve(F_form == 0, u, bc, J=J_form, solver_parameters=solver_params)
            
            # Обновление для следующего шага
            u_old.assign(u)
            last_converged_t = t
            
        except Exception as e:
            print(f"    Сходимость потеряна на шаге {step} (t={t:.1f} мс): {str(e)}")
            failed_step = step
            break
        
        t += dt_val
    
    return {
        'success': failed_step is None,
        'last_converged_time': last_converged_t,
        'failed_step': failed_step,
        'max_displacement': np.max(np.abs(u.vector().get_local())) if hasattr(np, 'max') else 0.0
    }

# Запуск варианта I с P1 и P2
results['I_P1'] = solve_variant_I(1)
results['I_P2'] = solve_variant_I(2)

# ============================================================================
# ВАРИАНТ II: Смешанная формулировка (u, p)
# ============================================================================
print("\n" + "="*60)
print("ВАРИАНТ II: Смешанная формулировка")
print("="*60)

def solve_variant_II(u_order, p_order):
    """
    Решение варианта II с заданными порядками элементов
    u_order: порядок для u (1 или 2)
    p_order: порядок для p (1)
    """
    print(f"\nВариант II с P{u_order}-P{p_order} элементами:")
    
    # Создание смешанного функционального пространства
    P_u = VectorElement("CG", mesh.ufl_cell(), u_order)
    P_p = FiniteElement("CG", mesh.ufl_cell(), p_order)
    
    W = FunctionSpace(mesh, P_u * P_p)
    
    # Функции для решения
    w = Function(W)  # (u, p)
    w_old = Function(W)  # предыдущее решение
    
    # Разделение функций
    u, p = split(w)
    u_old, p_old = split(w_old)
    
    # Начальные условия
    w_old.vector()[:] = 0.0
    w.vector()[:] = 0.0
    
    # Граничные условия (только для u на BASE)
    bc = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), bdry, BASE)
    
    # Пробные и тестовые функции
    (v, q) = TestFunctions(W)
    dw = TrialFunction(W)
    
    # Кинематические величины
    I = Identity(3)
    F = I + grad(u)
    J = det(F)
    
    # Тензор деформаций Грина-Лагранжа
    C = F.T * F
    E = (C - I) / 2.0
    
    # Q = b * tr(E²) = b * inner(E, E)
    Q = b * inner(E, E)
    
    # Энергия деформации
    psi_e = mu/2.0 * (exp(Q) - 1.0)
    psi_p = -p * J  # для смешанной формулировки
    
    # Первый тензор Пиолы-Кирхгоффа
    P_elastic = diff(psi_e, F)
    P_pressure = -p * J * inv(F).T
    P = P_elastic + P_pressure
    
    # Слабая форма:
    # 1. Уравнение равновесия: ∫P:∇v dx - граничные условия
    # 2. Условие несжимаемости: ∫q(J-1) dx = 0
    
    t_current = 0.0
    p_current = p_0(t_current)
    
    # adj(F) = J * F^{-T}
    adjF = J * inv(F).T
    
    # Форма для уравнения равновесия
    F_equilibrium = inner(P, grad(v)) * dx
    
    # Граничное условие на эндокарде
    F_equilibrium += dot(p_current * adjF.T * FacetNormal(mesh), v) * ds(ENDO)
    
    # Форма для условия несжимаемости
    F_incompressibility = q * (J - 1.0) * dx
    
    # Общая слабая форма
    F_form = F_equilibrium + F_incompressibility
    
    # Производная для метода Ньютона
    J_form = derivative(F_form, w, dw)
    
    # Параметры решателя Ньютона
    solver_params = {'newton_solver': {
        'linear_solver': 'mumps',  # для смешанных задач лучше mumps
        'absolute_tolerance': 1e-8,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 50,
        'relaxation_parameter': 1.0
    }}
    
    # Временной цикл
    t = 0.0
    max_time_steps = int(T / dt_val) + 1
    failed_step = None
    last_converged_t = 0.0
    
    for step in range(max_time_steps):
        if t > T + 1e-10:
            break
            
        print(f"  Шаг {step}: t = {t:.1f} мс, давление = {p_current:.2f} кПа")
        
        # Обновление давления на границе
        p_current = p_0(t)
        
        # Переопределение формы с новым давлением
        F_equilibrium = inner(P, grad(v)) * dx + dot(p_current * adjF.T * FacetNormal(mesh), v) * ds(ENDO)
        F_form = F_equilibrium + F_incompressibility
        J_form = derivative(F_form, w, dw)
        
        try:
            # Решение нелинейной задачи методом Ньютона
            solve(F_form == 0, w, bc, J=J_form, solver_parameters=solver_params)
            
            # Обновление для следующего шага
            w_old.assign(w)
            last_converged_t = t
            
        except Exception as e:
            print(f"    Сходимость потеряна на шаге {step} (t={t:.1f} мс): {str(e)}")
            failed_step = step
            break
        
        t += dt_val
    
    return {
        'success': failed_step is None,
        'last_converged_time': last_converged_t,
        'failed_step': failed_step,
        'max_displacement': np.max(np.abs(w.split()[0].vector().get_local())) if hasattr(np, 'max') else 0.0
    }

# Запуск варианта II с P2-P1 и P1-P1
try:
    results['II_P2-P1'] = solve_variant_II(2, 1)
    results['II_P1-P1'] = solve_variant_II(1, 1)
except Exception as e:
    print(f"Ошибка при решении варианта II: {e}")
    if 'II_P2-P1' not in results:
        results['II_P2-P1'] = {'success': False, 'failed_step': 0}
    if 'II_P1-P1' not in results:
        results['II_P1-P1'] = {'success': False, 'failed_step': 0}

# ============================================================================
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*60)

print("\nСводка по всем постановкам:")
print("-"*60)
print(f"{'Постановка':<15} {'Успех':<10} {'Последний шаг':<15} {'Причина остановки':<20}")
print("-"*60)

for name, result in results.items():
    success = result['success']
    last_t = result['last_converged_time']
    failed = result.get('failed_step', None)
    
    if success:
        status = "ДА"
        stop_reason = f"Достигнут t={T} мс"
    else:
        status = "НЕТ"
        stop_reason = f"Шаг {failed}, t={last_t:.1f} мс"
    
    print(f"{name:<15} {status:<10} {last_t:<15.1f} {stop_reason:<20}")

print("\n" + "="*60)
print("ВЫВОДЫ:")
print("="*60)

# Определение наиболее эффективной и надежной постановки
successful_formulations = [name for name, res in results.items() if res['success']]

if successful_formulations:
    print(f"Успешно завершились следующие постановки: {', '.join(successful_formulations)}")
    
    if len(successful_formulations) > 1:
        print("Для сравнения эффективности (приблизительная оценка по порядку элементов):")
        print("1. P1 элементы быстрее, но менее точны")
        print("2. P2 элементы медленнее, но точнее")
        print("3. Смешанные формулировки (II) требуют решения больших систем")
        
    # Рекомендация
    if 'II_P2-P1' in successful_formulations:
        print("\nРекомендуемая постановка: II с P2-P1 элементами")
        print("Причина: Оптимальное соотношение точности и стабильности для несжимаемых материалов")
    elif 'I_P2' in successful_formulations:
        print("\nРекомендуемая постановка: I с P2 элементами")
        print("Причина: Хорошая точность с приемлемой вычислительной стоимостью")
    else:
        print(f"\nРекомендуемая постановка: {successful_formulations[0]}")
        print("Причина: Единственная успешная постановка")
else:
    print("Ни одна из постановок не достигла конечного времени T=20 мс")
    print("Анализ точек сходимости:")
    
    # Сортировка по последнему успешному шагу
    sorted_results = sorted(results.items(), key=lambda x: x[1]['last_converged_time'], reverse=True)
    
    for name, result in sorted_results:
        print(f"  {name}: дошел до t={result['last_converged_time']:.1f} мс")

print("\n" + "="*60)
print("ЗАМЕЧАНИЯ ПО СХОДИМОСТИ:")
print("="*60)
print("1. Вариант I (штраф): Может иметь проблемы при больших κ из-за жесткости")
print("2. Вариант II (смешанный): Требует удовлетворения условию inf-sup (LBB)")
print("3. P1-P1 пара может не удовлетворять условию inf-sup для несжимаемых задач")
print("4. P2-P1 пара обычно удовлетворяет условию inf-sup")
print("5. Сходимость Ньютона зависит от начального приближения и шага по времени")