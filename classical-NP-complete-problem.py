import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Муравьиный алгоритм
# Муравьиный алгоритм – это метаэвристический алгоритм для решения задач оптимизации, инспирированный поведением муравьев при поиске путей к источникам пищи. 
# используем для нахождения оптимального пути в задаче коммивояжера

class AntColony:
    def __init__(self, graph, num_ants, alpha=1, beta=3, evaporation=0.5, pheromone_deposit=1):
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromone_deposit = pheromone_deposit
        self.num_vertices = len(graph)
        self.pheromone = np.ones((self.num_vertices, self.num_vertices))
        np.fill_diagonal(self.pheromone, 0)
        
        # Проверка на нулевые веса
        self.graph[self.graph == 0] = 1  # Если вес нулевой, заменяем его на 1

    def _select_next_vertex(self, current_vertex, visited):
        unvisited = [vertex for vertex in range(self.num_vertices) if vertex not in visited]
        probabilities = []
        for vertex in unvisited:
            weight = self.graph[current_vertex][vertex]
            if weight == 0:
                weight = 1  # Заменяем нулевой вес на 1
            pheromone = self.pheromone[current_vertex][vertex] ** self.alpha
            distance = 1 / weight ** self.beta  # Обратный вес для учета минимизации
            probabilities.append(pheromone * distance)
        probabilities /= np.sum(probabilities)
        next_vertex = np.random.choice(unvisited, p=probabilities)
        return next_vertex

    def _update_pheromone(self, ants):
        pheromone_delta = np.zeros((self.num_vertices, self.num_vertices))
        for ant, distance in ants:
            for i in range(len(ant) - 1):
                pheromone_delta[ant[i]][ant[i + 1]] += self.pheromone_deposit / distance
            pheromone_delta[ant[-1]][ant[0]] += self.pheromone_deposit / distance
        self.pheromone = (1 - self.evaporation) * self.pheromone + pheromone_delta

    def run(self, iterations):
        best_path = None
        best_distance = float('inf')
        for _ in range(iterations):
            ants = []
            for _ in range(self.num_ants):
                current_vertex = random.randint(0, self.num_vertices - 1)
                visited = [current_vertex]
                distance = 0
                while len(visited) < self.num_vertices:
                    next_vertex = self._select_next_vertex(current_vertex, visited)
                    distance += self.graph[current_vertex][next_vertex]
                    visited.append(next_vertex)
                    current_vertex = next_vertex
                distance += self.graph[visited[-1]][visited[0]]
                ants.append((visited, distance))
                if distance < best_distance:
                    best_distance = distance
                    best_path = visited
            self._update_pheromone(ants)
        return best_path, best_distance

# Метод отжига (Simulated Annealing)
# Метод отжига – это вероятностный алгоритм поиска экстремума в пространстве состояний. 
# используем для задачи коммивояжера

class SimulatedAnnealing:
    def __init__(self, graph, initial_temperature=100, cooling_rate=0.99, num_iterations=1000):
        self.graph = graph
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations

    def _get_initial_solution(self):
        return random.sample(range(len(self.graph)), len(self.graph))

    def _get_next_solution(self, current_solution):
        new_solution = current_solution[:]
        index1, index2 = random.sample(range(len(self.graph)), 2)
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]
        return new_solution

    def _acceptance_probability(self, energy, new_energy, temperature):
        if new_energy < energy:
            return 1
        return np.exp((energy - new_energy) / temperature)

    def run(self):
        current_solution = self._get_initial_solution()
        current_energy = self._calculate_energy(current_solution)
        best_solution = current_solution
        best_energy = current_energy
        temperature = self.initial_temperature
        for _ in range(self.num_iterations):
            new_solution = self._get_next_solution(current_solution)
            new_energy = self._calculate_energy(new_solution)
            if self._acceptance_probability(current_energy, new_energy, temperature) > random.random():
                current_solution = new_solution
                current_energy = new_energy
            if new_energy < best_energy:
                best_solution = new_solution
                best_energy = new_energy
            temperature *= self.cooling_rate
        return best_solution, best_energy

    def _calculate_energy(self, solution):
        energy = 0
        for i in range(len(solution)):
            energy += self.graph[solution[i - 1]][solution[i]]
        return energy

# Генерация графов разной плотности и сравнение алгоритмов

def gen_random_graph(V, E):
    """Generate an undirected graph in the form of an adjacency matrix with no duplicate edges or self loops"""
    graph = np.zeros((V, V))
    edges = random.sample([(i, j) for i in range(V) for j in range(i+1, V)], E)
    for u, v in edges:
        weight = random.randint(1, 100)
        graph[u][v] = weight
        graph[v][u] = weight
    return graph

def plot_graph(graph, title):
    plt.figure()
    plt.title(title)
    G = nx.Graph()

    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph[i][j] > 0:
                G.add_edge(i, j, weight=graph[i][j])

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_size=700)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()

def main():
    num_graphs = 1
    graph_sizes = [5, 20, 50]
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    total_idx = 0
    
    for size in graph_sizes:
        for density in densities:
            num_edges = int((size * (size - 1) / 2) * density)
            graphs = [gen_random_graph(size, num_edges) for _ in range(num_graphs)]
            
            for idx, graph in enumerate(graphs):
                total_idx += 1
                print(f"\n\tGraph {total_idx} (Size: {size}, Density: {density})")
                print("Minimum Spanning Tree:")
                
                print("\tAnt Colony Optimization:")
                ant_colony = AntColony(graph, num_ants=20)
                best_path_aco, best_distance_aco = ant_colony.run(iterations=100)
                print("Best Path:", best_path_aco)
                print("Best Distance:", best_distance_aco)
                
                print("\tSimulated Annealing:")
                sim_annealing = SimulatedAnnealing(graph)
                best_path_sa, best_distance_sa = sim_annealing.run()
                print("Best Path:", best_path_sa)
                print("Best Distance:", best_distance_sa)
                
                plot_graph(graph, f"Graph {total_idx} (Size: {size}, Density: {density})")

if __name__ == "__main__":
    main()




# def random_pick(n, m):
#     """Pick m integers from a bag of the integers in [0, n) without replacement"""
#     d = {i : i for i in range(m)} # For now, just pick the first m integers
#     res = []
#     for i in range(m): # Pick the i-th number
#         j = random.randrange(i, n)
#         # Pick whatever is in the j-th slot. If there is nothing, then pick j.
#         if j not in d:
#             d[j] = j
#         d[i], d[j] = d[j], d[i] # Swap the contents of the i-th and j-th slot
#         res.append(d[i])
#     return res

# def gen_random_graph(V, E):
#     """Generate an undirected graph in the form of an adjacency list with no duplicate edges or self loops"""
#     g = [[] for _ in range(V)]
#     edges = random_pick(math.comb(V, 2), E) # Pick E integers that represent the edges
#     for e in edges: # Decode the edges into their vertices
#         u = int((1 + math.sqrt(1 + 8 * e)) / 2)
#         v = e - math.comb(u, 2)
#         g[u].append(v)
#         g[v].append(u)
#     return g

# # The complete graph on 4 vertices
# print(gen_random_graph(4, 6)) # [[3, 2, 1], [3, 2, 0], [1, 0, 3], [0, 1, 2]]