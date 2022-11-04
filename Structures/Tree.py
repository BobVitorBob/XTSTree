class TreeNode:
  pass
class TreeNode:

	def __init__(self, content):
		self._left = None
		self._right = None
		self.tree = None
		self.cont = content

	@property
	def left():
		return self._left

	@left.setter
	def left(node: TreeNode):
		if self._left == None:
			self.tree._grow()

		self._left = node
		self._left.tree = self.tree
  
	@property
	def right():
		return self._right

	@right.setter
	def right(node: TreeNode):
		if self._right == None:
			self.tree._grow()

		self._right = node
		self._right.tree = self.tree

	
class Tree:

	def __init__(self):
		self.len = 0
	
	@property
	def root():
		return self._root

	@root.setter
	def root(node: TreeNode):
		self.len += 1
		self._root = node
		self._root._set_tree(self)
	
	def _grow():
		self.len += 1