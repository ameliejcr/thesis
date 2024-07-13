import math
import random
#from node_connections import NodeConnectionCollection
import excel_stuff
from typing import List, Optional

class Model:
    def __init__(self, bottomNodesCount: int, middleNodesCount: int, topNodesCount: int):
        self.bottom = Layer(bottomNodesCount)
        self.middle = Layer(middleNodesCount)
        self.top = Layer(topNodesCount)
        self.weights = NodeConnectionCollection()
        self.weights.create_connections(self.bottom.nodes, self.middle.nodes, 'bm')
        self.weights.create_connections(self.middle.nodes, self.top.nodes, 'mt')
        self.weights.build_index()

    def process(self, input_phon: tuple = None, input_sem: tuple = None):
        if input_phon and input_sem is not None:
            for i in range(len(self.bottom.nodes)):
                combined_input = input_phon + input_sem
                self.bottom.nodes[i].excitation = combined_input[i]
        elif input_phon is None:
            for i in range(144, 544):
                self.bottom.nodes[i].excitation = input_sem[(i-144)]
        else:
            for i in range(0, 144):
                self.bottom.nodes[i].excitation = input_phon[i]
        self.run_phases()
        if input_phon is None:
            for i in range(0, 144):
                if self.bottom.nodes[i].excitation == excel_stuff.phon_lib_test[i]:
                    print('-')
            else:
                print(False)
        elif input_sem is None:
            for i in range(144, 544):
                if self.bottom.nodes[i].excitation == excel_stuff.sem_lib_test[(i-144)]:
                    print('-')
            else:
                print(False)
        else:
            print('Done')


    def run_phases(self):
        self.initialSettlingPhase()
        self.hebbianLearningPhase()
        self.dreamingPhase()
        self.antiHebbianLearningPhase()

    def initialSettlingPhase(self):
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

    def sigma(self, value):
        return 1 / (1 + math.exp(-value))

    def hebbianLearningPhase(self):
        learningRate = 0.001
        for bottomNode in self.bottom.nodes:
            bottomNode.bias += bottomNode.excitation * learningRate
        for middleNode in self.middle.nodes:
            middleNode.bias += middleNode.excitation * learningRate
        for topNode in self.top.nodes:
            topNode.bias += topNode.excitation * learningRate
        for node in self.middle.nodes:
            for connection in self.weights.get_connections(node):
                connection.weight += learningRate * connection.node1.excitation * connection.node2.excitation

    def dreamingPhase(self):
        for i in range(10):
            for bottomNode in self.bottom.nodes:
                total = 0.0
                for connection in self.weights.get_connections(bottomNode):
                    total += connection.other_node(bottomNode).excitation * connection.weight
                bottomNode.excitation = bottomNode.bias + total

            for topNode in self.top.nodes:
                if topNode.excitation > random.random():
                    topNode.excitation = 1
                else:
                    topNode.excitation = 0

            for middleNode in self.middle.nodes:
                if middleNode.excitation > random.random():
                    middleNode.excitation = 1
                else:
                    middleNode.excitation = 0

    def antiHebbianLearningPhase(self):
        learningRate = 0.001
        for bottomNode in self.bottom.nodes:
            bottomNode.bias -= bottomNode.excitation * learningRate
        for middleNode in self.middle.nodes:
            middleNode.bias -= middleNode.excitation * learningRate
        for topNode in self.top.nodes:
            topNode.bias -= topNode.excitation * learningRate
        for node in self.middle.nodes:
            for connection in self.weights.get_connections(node):
                connection.weight -= learningRate * connection.node1.excitation * connection.node2.excitation



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
        self.node1 = node1
        self.node2 = node2
        self.weight = 0.0
        self.tag = ""

    def other_node(self, node: Node) -> Optional[Node]:
        if node == self.node1:
            return self.node2
        if node == self.node2:
            return self.node1
        return None

class NodeConnectionCollection:

    def __init__(self):
        self._items: List[NodeConnection] = list()
        self._index = {}

    def create_connections(self, layer1: List[Node], layer2: List[Node], tag: str):
        """
        Create a connection for each node in layer 1 to each node in layer 2
        """
        for node1 in layer1:
            for node2 in layer2:
                connection = NodeConnection(node1, node2)
                connection.tag = tag
                self._items.append(connection)

    def build_index(self):
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
        if node in self._index:
            if tag is None:
                return self._index[node]
            else:
                return [c for c in self._index[node] if c.tag == tag]
        return []

    def get_connection(self, node1: Node, node2: Node) -> NodeConnection:
        connections = self.get_connections(node1)
        return next(c for c in connections if (c.node2 == node2) or (c.node1 == node2))

    def set_weight(self, node1: Node, node2: Node, weight: int):
        self.get_connection(node1, node2).weight = weight
