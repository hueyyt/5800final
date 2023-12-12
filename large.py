import json
import random
import heapq
import time

def generate_random_graph(num_nodes, num_edges, max_weight):
    nodes = [str(i) for i in range(num_nodes)]
    edges = []

    for _ in range(num_edges):
        from_node = random.choice(nodes)
        to_node = random.choice(nodes)
        weight = random.randint(1, max_weight)
        edges.append({"from": from_node, "to": to_node, "weight": weight})

    return {"nodes": nodes, "edges": edges}

def save_graph_to_json(graph, filename):
    with open(filename, 'w') as file:
        json.dump(graph, file, indent=4)

def generate_and_save_graph(num_nodes, num_edges, max_weight, filename):
    graph = generate_random_graph(num_nodes, num_edges, max_weight)
    save_graph_to_json(graph, filename)


if __name__ == "__main__":
    num_nodes = 10
    num_edges = 300000
    max_weight = 200
    filename = 'random_graph_data.json'

    generate_and_save_graph(num_nodes, num_edges, max_weight, filename)


        
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def addNode(self, value):
        self.nodes.add(value)

    def addEdge(self, from_node, to_node, weight):
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node][to_node] = weight
    
    def dijkstra(self, start, end):
        shortest_paths = {node: float('infinity') for node in self.nodes}
        shortest_paths[start] = 0
        priority_queue = [(0, start)]
        predecessors = {node: None for node in self.nodes}

        while priority_queue:
            current_weight, current_node = heapq.heappop(priority_queue)

            if current_node == end:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = predecessors[current_node]
                return path[::-1], shortest_paths[end]

            for neighbor, weight in self.edges.get(current_node, {}).items():
                distance = current_weight + weight

                if distance < shortest_paths[neighbor]:
                    shortest_paths[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        return [], float('infinity')


class BiDirectionalDijkstra:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.forward = {node: float('infinity') for node in self.nodes}
        self.backward = {node: float('infinity') for node in self.nodes}
        self.forward_predecessors = {node: None for node in self.nodes}
        self.backward_predecessors = {node: None for node in self.nodes}
        self.visited_forward = set()  
        self.visited_backward = set()  

    def search(self, start, end):
        self.forward[start] = 0
        self.backward[end] = 0
        forward_queue = [(0, start)]
        backward_queue = [(0, end)]
        visited_forward = set()
        visited_backward = set()
        best_meeting_node = None
        best_distance = float('infinity')

        while forward_queue and backward_queue:
            self.expandNode(forward_queue, self.forward, self.forward_predecessors, visited_forward, True)
            self.expandNode(backward_queue, self.backward, self.backward_predecessors, visited_backward, False)

            # Check for meeting point
            meeting_node, distance = self.getMeetingNode(visited_forward, visited_backward)
            if meeting_node and distance < best_distance:
                best_meeting_node = meeting_node
                best_distance = distance

        if best_meeting_node:
            return self.constructPath(best_meeting_node, start, end), best_distance

        return [], float('infinity')

    def expandNode(self, queue, distances, predecessors, visited, is_forward):
        current_distance, current_node = heapq.heappop(queue)
        visited.add(current_node)

        if is_forward:
            neighbors = self.edges.get(current_node, {}).items()
        else:
            neighbors = [(node, weight[current_node]) for node, weight in self.edges.items() if current_node in weight]

        for neighbor, weight in neighbors:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    def getMeetingNode(self, visited_forward, visited_backward):
        best_node = None
        best_distance = float('infinity')
        for node in visited_forward:
            if node in visited_backward:
                distance = self.forward[node] + self.backward[node]
                if distance < best_distance:
                    best_node = node
                    best_distance = distance
        return best_node, best_distance

    def constructPath(self, meeting_node, start, end):
        path_forward = []
        path_backward = []

        # Construct forward path
        node = meeting_node
        while node != start:
            path_forward.append(node)
            node = self.forward_predecessors[node]
        path_forward.append(start)
        path_forward = path_forward[::-1]

        # Construct backward path
        node = meeting_node
        while node != end:
            node = self.backward_predecessors[node]
            path_backward.append(node)

        return path_forward + path_backward
    
    def getVisitedNodes(self):
        return self.visited_forward, self.visited_backward


with open('random_graph_data.json', 'r') as file:
    data = json.load(file)

graph = Graph()
for node in data['nodes']:
    graph.addNode(node)
for edge in data['edges']:
    graph.addEdge(edge['from'], edge['to'], edge['weight'])

start_node = '0'  
end_node = '99'   


start_time_standard = time.time()
shortest_path_standard, total_weight_standard = graph.dijkstra(start_node, end_node)
end_time_standard = time.time()
print("Standard Dijkstra's algorithm:")

print("Running time:", end_time_standard - start_time_standard)


bidirectional_dijkstra = BiDirectionalDijkstra(graph)
start_time_bi = time.time()
shortest_path_bi, total_weight_bi = bidirectional_dijkstra.search(start_node, end_node)
end_time_bi = time.time()
print("\nBi-Directional Dijkstra's algorithm:")

print("Running time:", end_time_bi - start_time_bi)

visited_forward, visited_backward = bidirectional_dijkstra.getVisitedNodes()
