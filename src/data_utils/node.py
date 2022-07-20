"""
Objet regroupant les caractéristiques d'un noeud du Branch and Bound.
Contient les noeuds fils.
"""

class BBNode:
    def __init__(self, bb_name: str, node_number: int, features: dict, value: float):
        self.bb_name = bb_name
        self.node_number = node_number
        self.features = features
        self.value = value

        self.parent_node = None
        self.children_nodes = []

    def add_child(self, child_node):
        self.children_nodes.append(child_node)
        child_node.parent_node = self

    def __str__(self):
        desc = ""
        parent_id = 'None'
        if self.parent_node is None:
            desc += self.bb_name + ':\n'
        else:
            parent_id = self.parent_node.node_number

        desc += f'[{parent_id}] -> [{self.node_number}]: {self.value}\n'
        desc += ''.join([str(child) for child in self.children_nodes])
        return desc