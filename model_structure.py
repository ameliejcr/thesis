import math
import random
from typing import List


class Model:
    def __init__(self,
                 bottom_nodes_count: int,
                 middle_nodes_count: int,
                 top_nodes_count: int,
                 phonetic_length: int):
        self.bottom = Layer(bottom_nodes_count)
        self.middle = Layer(middle_nodes_count)
        self.top = Layer(top_nodes_count)
        self.weights = NodeConnectionCollection()
        self.weights.create_connections(self.bottom.nodes, self.middle.nodes, 'bm')
        self.weights.create_connections(self.middle.nodes, self.top.nodes, 'mt')
        self.weights.rebuild_index()
        self.phonetic_length: int = phonetic_length

    def train(self, input_phon: tuple, input_sem: tuple):

        for i in range(len(self.bottom.nodes)):
            if i < self.phonetic_length:
                self.bottom.nodes[i].excitation = input_phon[i]
            else:
                self.bottom.nodes[i].excitation = input_sem[i - self.phonetic_length]

        self.run_phases()

    def predict(self, value: tuple, is_phonetic: bool, expected: tuple) -> float:
        if is_phonetic:
            for i in range(0, self.phonetic_length):
                self.bottom.nodes[i].excitation = value[i]
        else:
            for i in range(self.phonetic_length, len(self.bottom.nodes)):
                self.bottom.nodes[i].excitation = value[i - self.phonetic_length]

        self.run_phases()

        correct = 0
        if not is_phonetic:
            for i in range(0, self.phonetic_length):
                if self.bottom.nodes[i].excitation == expected[i]:
                    correct += 1
        else:
            for i in range(self.phonetic_length, len(self.bottom.nodes)):
                if self.bottom.nodes[i].excitation == expected[i - self.phonetic_length]:
                    correct += 1

        return correct / len(expected)

    def run_phases(self):
        self.initial_settling_phase()
        self.hebbian_learning_phase()
        self.dreaming_phase()
        self.anti_hebbian_learning_phase()

    def initial_settling_phase(self):
        for i in range(10):
            for middleNode in self.middle.nodes:
                total: float = 0.0
                for connection in self.weights.get_connections(middleNode):
                    total += connection.other_node(middleNode).excitation * connection.weight
                middleNode.excitation = self.sigma(middleNode.bias + total)

            for topNode in self.top.nodes:
                total = 0.0
                for connection in self.weights.get_connections(topNode, 'mt'):
                    total += connection.other_node(topNode).excitation * connection.weight
                topNode.excitation = self.sigma(topNode.bias + total)

    @staticmethod
    def sigma(value):
        return 1 / (1 + math.exp(-value))

    def hebbian_learning_phase(self):
        learning_rate = 0.001
        for bottomNode in self.bottom.nodes:
            bottomNode.bias += bottomNode.excitation * learning_rate
        for middleNode in self.middle.nodes:
            middleNode.bias += middleNode.excitation * learning_rate
        for topNode in self.top.nodes:
            topNode.bias += topNode.excitation * learning_rate
        for node in self.middle.nodes:
            for connection in self.weights.get_connections(node):
                connection.weight += learning_rate * connection.node1.excitation * connection.node2.excitation

    def dreaming_phase(self):
        for i in range(10):
            for bottomNode in self.bottom.nodes:
                total = 0.0
                for connection in self.weights.get_connections(bottomNode):
                    total += connection.other_node(bottomNode).excitation * connection.weight
                bottomNode.excitation = bottomNode.bias + total

            for topNode in self.top.nodes:
                total = 0.0
                for connection in self.weights.get_connections(topNode):
                    total += connection.other_node(topNode).excitation * connection.weight
                topNode.excitation = self.bernoulli_distribution(self.sigma(topNode.bias + total))

            for middleNode in self.middle.nodes:
                total: float = 0.0
                for connection in self.weights.get_connections(middleNode):
                    total += connection.other_node(middleNode).excitation * connection.weight
                middleNode.excitation = self.bernoulli_distribution(self.sigma(middleNode.bias + total))

    def anti_hebbian_learning_phase(self):
        learning_rate = 0.001
        for bottomNode in self.bottom.nodes:
            bottomNode.bias -= bottomNode.excitation * learning_rate
        for middleNode in self.middle.nodes:
            middleNode.bias -= middleNode.excitation * learning_rate
        for topNode in self.top.nodes:
            topNode.bias -= topNode.excitation * learning_rate
        for node in self.middle.nodes:
            for connection in self.weights.get_connections(node):
                connection.weight -= learning_rate * connection.node1.excitation * connection.node2.excitation

    def remove_random_nodes(self, percentage: float):
        if (percentage < 0) or (percentage > 1):
            raise Exception("percentage must be a fraction.")

        layers = [self.bottom, self.middle, self.top]
        nodes_to_remove = []
        for layer in layers:
            for node in layer.nodes:
                if random.random() <= percentage:
                    layer.nodes.remove(node)
                    nodes_to_remove.append(node)

        self.weights.remove_nodes(nodes_to_remove)

    @staticmethod
    def bernoulli_distribution(probability: float) -> float:
        if probability > random.random():
            return 1.0
        else:
            return 0.0


class Layer:
    def __init__(self, count: int):
        self.nodes = list[Node]()
        for item in range(count):
            self.nodes.append(Node())


class Node:
    def __init__(self):
        self.excitation: float = 0.0
        self.bias: float = 0.0


class NodeConnection:
    def __init__(self, node1: Node, node2: Node):
        self.node1: Node = node1
        self.node2: Node = node2
        self.weight = 0.0
        self.tag = ""

    def other_node(self, node: Node) -> Node:
        if node is self.node1:
            return self.node2
        else:
            return self.node1


class NodeConnectionCollection:

    def __init__(self):
        self._items: List[NodeConnection] = list()
        self._index = None

    def create_connections(self, layer1: List[Node], layer2: List[Node], tag: str):
        for node1 in layer1:
            for node2 in layer2:
                connection = NodeConnection(node1, node2)
                connection.tag = tag
                self._items.append(connection)

    def rebuild_index(self):
        self._index = {}
        for item in self._items:
            if item.node1 in self._index:
                self._index[item.node1].append(item)
            else:
                self._index[item.node1] = [item]
            if item.node2 in self._index:
                self._index[item.node2].append(item)
            else:
                self._index[item.node2] = [item]

    def get_connections(self, node: Node, tag: str = None) -> List[NodeConnection]:
        if self._index is None:
            self.rebuild_index()

        if node in self._index:
            if tag is None:
                return self._index[node]
            else:
                return [c for c in self._index[node] if c.tag == tag]
        return []

    def get_connection(self, node1: Node, node2: Node) -> NodeConnection:
        connections = self.get_connections(node1)
        return next(c for c in connections if (c.node2 == node2) or (c.node1 == node2))

    def remove_nodes(self, nodes: List[Node]):
        for node in nodes:
            self._items[:] = [x for x in self._items if x.node1 != node and x.node2 != node]

        self._index = None

    def set_weight(self, node1: Node, node2: Node, weight: int):
        self.get_connection(node1, node2).weight = weight
