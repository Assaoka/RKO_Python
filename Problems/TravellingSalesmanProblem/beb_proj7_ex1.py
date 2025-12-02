import gurobipy as gp
from gurobipy import GRB
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys

global_best_integer_value = -math.inf
global_best_integer_solution = None
tree_log = []
problem_name = ""

def solve_lp_relaxation(model_data, current_constraints):
    try:
        m = gp.Model("Relaxation")
        m.setParam('OutputFlag', 0)
        
        x = m.addVar(lb=0, name="x")
        y = m.addVar(lb=0, name="y")
        
        m.setObjective(model_data['fo']['x'] * x + model_data['fo']['y'] * y, GRB.MAXIMIZE)
        
        for idx, constr in enumerate(model_data['restricoes']):
            m.addConstr(constr['x'] * x + constr['y'] * y <= constr['rhs'], f"R{idx}")
            
        for var, bound_type, value in current_constraints:
            if var == 'x':
                if bound_type == '<=':
                    m.addConstr(x <= value, f"B_X_LE_{value}")
                else:
                    m.addConstr(x >= value, f"B_X_GE_{value}")
            elif var == 'y':
                if bound_type == '<=':
                    m.addConstr(y <= value, f"B_Y_LE_{value}")
                else:
                    m.addConstr(y >= value, f"B_Y_GE_{value}")
        
        m.optimize()
        print(m.status)
        
        if m.status == GRB.OPTIMAL:
            return m.objVal, {'x': x.X, 'y': y.X}
        
        return -math.inf, None

    except gp.GurobiError:
        return -math.inf, None

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
    
    if lp_value <= global_best_integer_value + 1e-6 or lp_value == -math.inf:
        node_info['action'] = 'Podado'
        return
    
    x_val = lp_solution['x']
    y_val = lp_solution['y']
    
    x_is_int = abs(x_val - round(x_val)) < 1e-6
    y_is_int = abs(y_val - round(y_val)) < 1e-6
    
    if x_is_int and y_is_int:
        node_info['action'] = 'Podado (Sol. Inteira)'
        node_info['is_integer'] = True
        
        if lp_value > global_best_integer_value + 1e-6:
            global_best_integer_value = lp_value
            global_best_integer_solution = {'x': round(x_val), 'y': round(y_val)}
            node_info['new_incumbent'] = True
            
        return
        
    node_info['action'] = 'Ramificado'
    
    x_frac = abs(x_val - round(x_val))
    y_frac = abs(y_val - round(y_val))

    if x_frac > 1e-6 and x_frac >= y_frac:
        var_to_branch = 'x'
        val_to_branch = x_val
    elif y_frac > 1e-6:
        var_to_branch = 'y'
        val_to_branch = y_val
    else:
        return

    floor_val = math.floor(val_to_branch)
    ceil_val = math.ceil(val_to_branch)
    
    constr_le = current_constraints + [(var_to_branch, '<=', floor_val)]
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
        
        if node['lp_value'] == -math.inf:
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
    global_best_integer_value = -math.inf
    global_best_integer_solution = None
    tree_log = []
    
    print(f"\n======== PROBLEMA {p_key.upper()} - INÍCIO DO B&B ========")
    
    branch_and_bound([], 0, 1, "depth")
            
    print("\n---------------- RESULTADO FINAL ----------------")
    print(f"Melhor Valor Inteiro (Z*): {global_best_integer_value:,.2f}")
    print(f"Solução Ótima Inteira (x, y): {global_best_integer_solution}")
    print("=================================================")
    
    try:
        plot_bnb_tree(tree_log, p_key)
    except Exception as e:
        print(f"\nAVISO: Falha ao plotar o gráfico. Certifique-se de que 'networkx' e 'matplotlib' estão instalados.")

PROBLEMS_DATA = {
    'a': {
        'fo': {'x': 5, 'y': 2},
        'restricoes': [
            {'x': 3, 'y': 1, 'rhs': 12},
            {'x': 1, 'y': 1, 'rhs': 5}
        ]
    },
    'b': {
        'fo': {'x': 2, 'y': 3},
        'restricoes': [
            {'x': 1, 'y': 2, 'rhs': 10},
            {'x': 3, 'y': 4, 'rhs': 25}
        ]
    },
    'c': {
        'fo': {'x': 4, 'y': 3},
        'restricoes': [
            {'x': 4, 'y': 9, 'rhs': 26},
            {'x': 8, 'y': 5, 'rhs': 17}
        ]
    },
    'd': {
        'fo': {'x': 1, 'y': 1},
        'restricoes': [
            {'x': 2, 'y': 2, 'rhs': 3},
            {'x': 7, 'y': 3, 'rhs': 22}
        ]
    }
}

if __name__ == '__main__':
    
    run_problem_and_log('a')
    run_problem_and_log('b')
    run_problem_and_log('c')
    run_problem_and_log('d')