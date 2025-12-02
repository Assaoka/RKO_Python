import gurobipy as gp
from gurobipy import GRB
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys

global_best_integer_value = math.inf
global_best_integer_solution = None
tree_log = []
problem_name = ""

def solve_lp_relaxation(model_data, current_constraints):
    try:
        m = gp.Model("Relaxation")
        m.setParam('OutputFlag', 0)
        
        I = model_data['armazens']
        J = model_data['clientes']
        
        # Variáveis de transporte (Contínuas)
        x = m.addVars(I, J, lb=0, name="x")
        # Variáveis de abertura (Binárias relaxadas, lb=0, ub=1)
        y = m.addVars(I, lb=0, ub=1, name="y")
        
        # FO: Minimizar (Renda Fixa) + (Custo Variável de Transporte)
        custo_fixo = gp.quicksum(model_data['renda'][i] * y[i] for i in I)
        custo_transporte = gp.quicksum(model_data['custo_unitario'][i, j] * x[i, j] for i in I for j in J)
        m.setObjective(custo_fixo + custo_transporte, GRB.MINIMIZE)
        
        # 1. Satisfação da Demanda
        for j in J:
            m.addConstr(gp.quicksum(x[i, j] for i in I) == model_data['demanda'][j], f"R_Demanda_{j}")
            
        # 2. Restrição de Ligação (Big M / Capacidade)
        for i in I:
            m.addConstr(gp.quicksum(x[i, j] for j in J) <= model_data['capacidade'][i] * y[i], f"R_Capacidade_Link_{i}")
            
        # Restrições de Branching (Aplicadas apenas a Y)
        for var_name, bound_type, value in current_constraints:
            i = var_name[2]
            gurobi_var = y[i] 
            
            if bound_type == '<=':
                m.addConstr(gurobi_var <= value, f"B_Y_LE_{i}_{value}")
            else:
                m.addConstr(gurobi_var >= value, f"B_Y_GE_{i}_{value}")
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            y_sol = {i: y[i].X for i in I}
            x_sol = {key: x[key].X for key in x.keys()}
            return m.objVal, {'y': y_sol, 'x': x_sol}
        
        return math.inf, None

    except gp.GurobiError:
        return math.inf, None

def branch_and_bound(current_constraints, parent_id, node_id, strategy="depth"):
    global global_best_integer_value
    global global_best_integer_solution
    global tree_log
    
    lp_value, lp_solution = solve_lp_relaxation(PROBLEMS_DATA[problem_name], current_constraints)

    node_info = {
        'id': node_id, 
        'parent': parent_id, 
        'lp_value': lp_value, 
        'lp_solution': lp_solution, 
        'action': 'Aberto',
        'is_integer': False,
        'new_incumbent': False
    }
    tree_log.append(node_info)
    
    # Poda por Bound (Minimiza!)
    if lp_value >= global_best_integer_value - 1e-6 or lp_value == math.inf:
        node_info['action'] = 'Podado'
        return
    
    y_sol = lp_solution['y']
    
    is_integer = True
    var_to_branch = None
    val_to_branch = 0
    max_frac = -1

    # Encontra a variável y mais fracionária
    for i, y_val in y_sol.items():
        y_is_int = abs(y_val - round(y_val)) < 1e-6
        if not y_is_int:
            is_integer = False
            frac = abs(y_val - round(y_val))
            if frac > max_frac:
                max_frac = frac
                var_to_branch = f'y_{i}'
                val_to_branch = y_val
                
    if is_integer:
        node_info['action'] = 'Podado (Sol. Inteira)'
        node_info['is_integer'] = True
        
        if lp_value < global_best_integer_value - 1e-6:
            global_best_integer_value = lp_value
            global_best_integer_solution = {k: round(v) for k, v in y_sol.items()}
            node_info['new_incumbent'] = True
            
        return
        
    node_info['action'] = 'Ramificado'
    
    floor_val = math.floor(val_to_branch)
    ceil_val = math.ceil(val_to_branch)
    
    # Branch 1: <= floor (Fixar em 0)
    constr_le = current_constraints + [(var_to_branch, '<=', floor_val)]
    # Branch 2: >= ceil (Fixar em 1)
    constr_ge = current_constraints + [(var_to_branch, '>=', ceil_val)]
    
    branch_and_bound(constr_le, node_id, node_id * 2 + 1, strategy)
    branch_and_bound(constr_ge, node_id, node_id * 2 + 2, strategy)

def calculate_tree_positions(tree_log):
    pos = {}
    levels = {1: 0}
    
    queue = [1]
    while queue:
        node_id = queue.pop(0)
        level = levels[node_id]
        
        children = [n['id'] for n in tree_log if n['parent'] == node_id]
        
        for child_id in children:
            if child_id not in levels:
                levels[child_id] = level + 1
                queue.append(child_id)

    max_level = max(levels.values())
    
    nodes_per_level = [[] for _ in range(max_level + 1)]
    for node in tree_log:
        level = levels.get(node['id'], -1)
        if level != -1:
            nodes_per_level[level].append(node['id'])

    for level, node_list in enumerate(nodes_per_level):
        num_nodes = len(node_list)
        if num_nodes == 0: continue
        
        x_step = 1.0 / (num_nodes + 1)
        
        for i, node_id in enumerate(node_list):
            x_pos = (i + 1) * x_step - 0.5
            y_pos = -level * 0.5 
            
            pos[node_id] = (x_pos, y_pos)

    return pos


def plot_bnb_tree(tree_log, p_key):
    G = nx.DiGraph()
    
    for node in tree_log:
        node_id = node['id']
        parent_id = node['parent']
        G.add_node(node_id)
        
        if parent_id != 0:
            G.add_edge(parent_id, node_id)
    
    pos = calculate_tree_positions(tree_log)
    
    labels = {}
    node_colors = []
    
    for node in tree_log:
        node_id = node['id']
        
        if node_id not in pos: continue

        z_val = f"Z={node['lp_value']:.2f}"
        
        if node['lp_value'] == math.inf:
             color = 'gray'
             labels[node_id] = f"Nó {node_id}\n(Inviável)"
        elif node['action'] == 'Podado':
            color = 'red'
            labels[node_id] = f"{z_val}\n(Podado)"
        elif node['new_incumbent']:
            color = 'lime'
            labels[node_id] = f"{z_val}\n(Ótimo Inteiro!)"
        elif node['is_integer']:
            color = 'yellow'
            labels[node_id] = f"{z_val}\n(Inteiro)"
        else:
            color = 'skyblue'
            labels[node_id] = z_val
            
        node_colors.append(color)

    plt.figure(figsize=(20, 10))
    
    nx.draw(G, pos, with_labels=False, node_size=2500, node_color=node_colors, font_size=10, font_weight='bold', arrowsize=15, cmap=plt.cm.Blues)
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title(f"Árvore Branch-and-Bound (Layout Hierárquico) para o Problema {p_key.upper()}")
    plt.show()

def run_problem_and_log(p_key):
    global global_best_integer_value
    global global_best_integer_solution
    global tree_log
    global problem_name
    
    problem_name = p_key
    global_best_integer_value = math.inf
    global_best_integer_solution = None
    tree_log = []
    
    print(f"\n======== PROBLEMA {p_key.upper()} - INÍCIO DO B&B ========")
    
    branch_and_bound([], 0, 1, "depth")
            
    print("\n---------------- RESULTADO FINAL ----------------")
    print(f"Melhor Valor Inteiro (Z*): {global_best_integer_value:,.2f}")
    print(f"Armazéns Abertos (y): {global_best_integer_solution}")
    print("=================================================")
    
    try:
        plot_bnb_tree(tree_log, p_key)
    except Exception as e:
        print(f"\nAVISO: Falha ao plotar o gráfico. Certifique-se de que 'networkx' e 'matplotlib' estão instalados.")

PROBLEMS_DATA = {
    'Q4': {
        'armazens': ['A', 'B', 'C', 'D'],
        'clientes': ['a', 'b', 'c', 'd', 'e'],
        'renda': {'A': 50, 'B': 32, 'C': 28, 'D': 36},
        'capacidade': {'A': 35, 'B': 28, 'C': 22, 'D': 28},
        'demanda': {'a': 14, 'b': 12, 'c': 10, 'd': 12, 'e': 8},
        'custo_unitario': {
            ('A', 'a'): 2, ('A', 'b'): 5, ('A', 'c'): 1, ('A', 'd'): 2, ('A', 'e'): 5,
            ('B', 'a'): 4, ('B', 'b'): 4, ('B', 'c'): 9, ('B', 'd'): 1, ('B', 'e'): 4,
            ('C', 'a'): 1, ('C', 'b'): 8, ('C', 'c'): 5, ('C', 'd'): 6, ('C', 'e'): 2,
            ('D', 'a'): 7, ('D', 'b'): 1, ('D', 'c'): 2, ('D', 'd'): 1, ('D', 'e'): 8
        }
    }
}

if __name__ == '__main__':
    
    run_problem_and_log('Q4')