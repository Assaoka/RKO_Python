import math

valores = [10.0, 15.0, 36.0, 20.0, 15.0, 18.0, 20.0]
custos = [10.0, 17.0, 49.0, 30.0, 11.0, 21.0, 31.0]
ratio = [v / c for v,c in zip(valores,custos)]
custo_max = 100.0

global_best_value = 0.0
global_best_int_value = 0.0
global_best_int_solution = [0] * len(valores)

def solve_relax(x):
    sol = x[:] 
    custo_atual = 0.0
    valor_atual = 0.0
    nulos_indices = []
    
    for i in range (len(x)):
        if x[i] is not None:
            custo_atual += custos[i]*x[i]
            valor_atual += valores[i]*x[i]
        else:
            nulos_indices.append(i)

    if custo_atual > custo_max + 1e-6:
        return 0.0, [0] * len(x)
    
    nulos_ordenados = sorted(
        nulos_indices,
        key=lambda i: ratio[i],
        reverse=True
    )
    
    for idx in nulos_ordenados:
        if custo_atual + custos[idx] <= custo_max + 1e-6:
            sol[idx] = 1.0
            custo_atual += custos[idx]
            valor_atual += valores[idx]
        else:
            frac = (custo_max - custo_atual)/custos[idx]
            sol[idx] = frac
            valor_atual += valores[idx] * frac
            custo_atual = custo_max
            break 
            
    for i in nulos_indices:
        if sol[i] is None:
            sol[i] = 0.0
    
    return valor_atual, sol


def beb(solution):
    global global_best_value
    global global_best_int_value
    global global_best_int_solution
    
    out = solve_relax(solution)

    if out[0] == 0.0:
        return 
    
    best_value, relaxed_solution = out

    if best_value <= global_best_int_value - 1e-6:
        return

    real_variable_idx = None
    for i in range(len(solution)):
        val = relaxed_solution[i]
        if val is not None and abs(val - round(val)) > 1e-6:
            real_variable_idx = i
            break

    if real_variable_idx is None:
        if best_value > global_best_int_value + 1e-6:
            global_best_int_value = best_value
            global_best_int_solution = [int(round(x)) for x in relaxed_solution]
            
            print("Novo melhor valor inteiro encontrado:")
            print(f"Valor: {global_best_int_value:.2f}")
            print(f"Solução: {global_best_int_solution}")
        return
    
    
    new_solution_0 = solution[:]
    new_solution_0[real_variable_idx] = 0
    beb(new_solution_0)
    
    new_solution_1 = solution[:]
    new_solution_1[real_variable_idx] = 1
    beb(new_solution_1)
    
    return


initial_solution = [None, None, None, None, None, None, None]
beb(initial_solution)

print("\n" + "=" * 50)
print(f"Solução Final do Problema da Mochila (B&B):")
print(f"Melhor Valor Inteiro: {global_best_int_value:.2f}")
print(f"Solução Ótima (x1..x7): {global_best_int_solution}")
print("=" * 50)