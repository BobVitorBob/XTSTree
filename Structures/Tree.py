class TreeNode:
  pass
class TreeNode:

  def __init__(self, content):
    self.left = None
    self.right = None
    self.cont = content

class Tree:

  def __init__(self):
    self.root = None
  
  def to_list(self):
    return Tree._sub_node_to_list(self.root)

  def get_leaves(self):
    return Tree._get_leaves(self.root)
      
  @staticmethod
  def _get_leaves(node: TreeNode):
    if node.left is None and node.right is None:
      return [node.cont]
    return [*Tree._get_leaves(node.left), *Tree._get_leaves(node.right)]

  @staticmethod
  def _sub_node_to_list(node: TreeNode):
    if node is None:
      return []
    return [*Tree._sub_node_to_list(node.left), node.cont, *Tree._sub_node_to_list(node.right)]

  def __len__(self):
    return Tree._count_nodes(self.root)
    
  @staticmethod
  def _count_nodes(node: TreeNode):
    if node is None:
      return 0
    return 1 + Tree._count_nodes(node.left) + Tree._count_nodes(node.right)
